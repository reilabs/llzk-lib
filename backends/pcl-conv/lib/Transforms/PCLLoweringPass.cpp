//===-- PCLLoweringPass.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-to-pcl` pass.
///
//===----------------------------------------------------------------------===//

#include "pcl-conv/Transforms/TransformationPasses.h"

#include "llzk/Config/Config.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Transforms/LLZKLoweringUtils.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/Field.h"

#include <pcl/Dialect/IR/Dialect.h>
#include <pcl/Dialect/IR/Ops.h>
#include <pcl/Dialect/IR/Types.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>

#include <deque>
#include <memory>

// Include the generated base pass class definitions.
namespace pcl::conversion {
#define GEN_PASS_DECL_PCLLOWERINGPASS
#define GEN_PASS_DEF_PCLLOWERINGPASS
#include "pcl-conv/Transforms/TransformationPasses.h.inc"
} // namespace pcl::conversion

using namespace mlir;
using namespace llzk;
using namespace llzk::cast;
using namespace llzk::boolean;
using namespace llzk::constrain;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;

namespace {

static FailureOr<Value> lookup(Value v, llvm::DenseMap<Value, Value> &m, Operation *onError) {
  if (auto it = m.find(v); it != m.end()) {
    return it->second;
  }
  return onError->emitError("missing operand mapping");
}

static void rememberResult(Value from, Value to, llvm::DenseMap<Value, Value> &m) {
  (void)m.try_emplace(from, to);
}

static std::optional<llvm::APInt> getPclConstAPInt(Value v) {
  if (auto c = llvm::dyn_cast_if_present<pcl::ConstOp>(v.getDefiningOp())) {
    // Chain: ConstOp -> FeltAttr (or BoolAttr-as-int) -> IntegerAttr -> APInt
    return c.getValue().getValue().getValue();
  }
  return std::nullopt;
}

// Convert binary LLZK op to corresponding binary PCL op
template <typename SrcBinOp, typename DstBinOp>
static LogicalResult
lowerBinaryLike(OpBuilder &b, SrcBinOp src, llvm::DenseMap<Value, Value> &mapping) {
  auto loc = src.getLoc();
  auto lhs = lookup(src.getLhs(), mapping, src);
  if (failed(lhs)) {
    return failure();
  }
  auto rhs = lookup(src.getRhs(), mapping, src);
  if (failed(rhs)) {
    return failure();
  }

  auto dst = b.create<DstBinOp>(loc, *lhs, *rhs);
  rememberResult(src.getResult(), dst.getRes(), mapping);
  return success();
}

// Convert unary LLZK op to corresponding unary PCL op
template <typename SrcBinOp, typename DstBinOp>
static LogicalResult
lowerUnaryLike(OpBuilder &b, SrcBinOp src, llvm::DenseMap<Value, Value> &mapping) {
  auto loc = src.getLoc();
  auto operand = lookup(src.getOperand(), mapping, src);
  if (failed(operand)) {
    return failure();
  }

  auto dst = b.create<DstBinOp>(loc, *operand);
  rememberResult(src.getResult(), dst.getRes(), mapping);
  return success();
}

static LogicalResult lowerConstImpl(
    OpBuilder &b, Value result, Location location, const llvm::APInt &value,
    const llvm::APInt &prime, llvm::DenseMap<Value, Value> &mapping
) {
  // FeltConstAttr does not enforce a canonical (< p) value, but the constant
  // fast paths downstream (literal bit vectors, power-of-two divisor checks,
  // modExp exponents) all read this op's APInt as the field element. Reduce
  // mod p so they never see an unreduced representative.
  llvm::APInt canonical = value;
  unsigned w = std::max(value.getBitWidth(), prime.getBitWidth());
  llvm::APInt vExt = value.zextOrTrunc(w);
  llvm::APInt pExt = prime.zextOrTrunc(w);
  if (vExt.uge(pExt)) {
    canonical = vExt.urem(pExt);
  }
  auto attr = pcl::FeltAttr::get(b.getContext(), canonical);
  auto dst = b.create<pcl::ConstOp>(location, attr);
  rememberResult(result, dst.getRes(), mapping);
  return success();
}

static LogicalResult lowerConst(
    OpBuilder &b, FeltConstantOp cst, const llvm::APInt &prime,
    llvm::DenseMap<Value, Value> &mapping
) {
  auto value = cst.getValue().getValue();
  return lowerConstImpl(b, cst.getResult(), cst->getLoc(), value, prime, mapping);
}

static LogicalResult lowerConst(
    OpBuilder &b, mlir::arith::ConstantOp cst, const llvm::APInt &prime,
    llvm::DenseMap<Value, Value> &mapping
) {
  auto value = mlir::cast<mlir::IntegerAttr>(cst.getValue()).getValue();
  return lowerConstImpl(b, cst.getResult(), cst.getLoc(), value, prime, mapping);
}

/// Bit-decomposes `pclValue` into `width` bits, three constraints:
///   1. Booleanity: `b_i · (b_i − 1) == 0` for each bit.
///   2. Weighted sum equality: `Σ b_i · 2^i == pclValue`.
///   3. Range check: `Σ b_i · 2^i < prime`, via a bitwise comparison against `prime`'s bits.
/// The range check is only needed at full field width: for `width <
/// prime.getActiveBits()` the sum is at most `2^width − 1 < prime` already.
/// Callers must only pass a narrow `width` for values known to be `< 2^width`,
/// since the sum equality then also asserts that bound.
/// Returns the bit vector with the low bit at index 0.
static SmallVector<Value> decomposeBits(
    OpBuilder &b, Location loc, Value pclValue, const llvm::APInt &prime,
    llvm::function_ref<std::string()> nameGen, unsigned width
) {
  assert(width >= 1 && width <= prime.getActiveBits() && "width must be in [1, bits(prime)]");
  auto *ctx = b.getContext();
  unsigned constBits = prime.getActiveBits() + 1;
  auto zeroConst = b.create<pcl::ConstOp>(loc, pcl::FeltAttr::get(ctx, llvm::APInt(constBits, 0)));
  auto oneConst = b.create<pcl::ConstOp>(loc, pcl::FeltAttr::get(ctx, llvm::APInt(constBits, 1)));

  SmallVector<Value> bits;
  bits.reserve(width);
  Value acc = zeroConst.getRes();
  llvm::APInt weight(constBits, 1);

  for (unsigned i = 0; i < width; ++i) {
    auto bit = b.create<pcl::VarOp>(loc, nameGen(), /*is_output=*/false);
    bits.push_back(bit.getRes());

    // Booleanity: b · (b − 1) == 0.
    auto bMinus1 = b.create<pcl::SubOp>(loc, bit.getRes(), oneConst.getRes());
    auto prod = b.create<pcl::MulOp>(loc, bit.getRes(), bMinus1.getRes());
    auto isZero = b.create<pcl::CmpEqOp>(loc, prod.getRes(), zeroConst.getRes());
    b.create<pcl::AssertOp>(loc, isZero.getRes());

    // Weighted sum: acc += b · 2^i.
    auto w = b.create<pcl::ConstOp>(loc, pcl::FeltAttr::get(ctx, weight));
    auto term = b.create<pcl::MulOp>(loc, bit.getRes(), w.getRes());
    acc = b.create<pcl::AddOp>(loc, acc, term.getRes()).getRes();
    weight <<= 1;
  }

  auto eq = b.create<pcl::CmpEqOp>(loc, acc, pclValue);
  b.create<pcl::AssertOp>(loc, eq.getRes());

  if (width < prime.getActiveBits()) {
    return bits;
  }

  // Range check: Σ b_i · 2^i < prime. Two
  // accumulators, each in {0, 1}:
  //   strictLess — the prefix seen so far is strictly less than p's prefix.
  //   stillEqual — the prefix seen so far equals p's prefix.
  // strictLess + stillEqual == 0 means the prefix already exceeded p; the
  // final `strictLess == 1` assert catches that case.
  Value strictLess = zeroConst.getRes();
  Value stillEqual = oneConst.getRes();
  for (int i = width - 1; i >= 0; --i) {
    Value bit = bits[i];
    auto complement = b.create<pcl::SubOp>(loc, oneConst.getRes(), bit);
    if (prime[i]) {
      auto delta = b.create<pcl::MulOp>(loc, stillEqual, complement.getRes());
      strictLess = b.create<pcl::AddOp>(loc, strictLess, delta.getRes()).getRes();
      stillEqual = b.create<pcl::MulOp>(loc, stillEqual, bit).getRes();
    } else {
      stillEqual = b.create<pcl::MulOp>(loc, stillEqual, complement.getRes()).getRes();
    }
  }
  auto ltEq = b.create<pcl::CmpEqOp>(loc, strictLess, oneConst.getRes());
  b.create<pcl::AssertOp>(loc, ltEq.getRes());

  return bits;
}

/// Combinator inverse of `decomposeBits`: given a bit vector (low bit first),
/// returns `Σ bits[i] · 2^i`. Emits no assertions. Callers are responsible for
/// any range-check or booleanity constraints on the input bits.
///
/// Precondition: `bits` is non-empty.
static Value recomposeBits(OpBuilder &b, Location loc, ArrayRef<Value> bits) {
  assert(!bits.empty() && "recomposeBits requires at least one bit");
  auto *ctx = b.getContext();
  unsigned constBits = static_cast<unsigned>(bits.size()) + 1;
  llvm::APInt weight(constBits, 1);

  Value acc = bits[0];
  weight <<= 1;

  for (unsigned i = 1, e = bits.size(); i < e; ++i) {
    auto w = b.create<pcl::ConstOp>(loc, pcl::FeltAttr::get(ctx, weight));
    auto term = b.create<pcl::MulOp>(loc, bits[i], w.getRes());
    acc = b.create<pcl::AddOp>(loc, acc, term.getRes()).getRes();
    weight <<= 1;
  }
  return acc;
}

class PassImpl : public pcl::conversion::impl::PCLLoweringPassBase<PassImpl> {
  using Base = PCLLoweringPassBase<PassImpl>;
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pcl::PCLDialect, func::FuncDialect>();
  }

  /// The translation only works now on LLZK structs where all the members are felts.
  LogicalResult validateStruct(StructDefOp structDef) {
    for (auto member : structDef.getMemberDefs()) {
      auto memberType = member.getType();
      if (!llvm::isa<FeltType>(memberType)) {
        return member.emitError() << "Member must be felt type. Found " << memberType
                                  << " for member: " << member.getName();
      }
    }
    return success();
  }

  /// Emit assertions for an equality `lhs == rhs`, with fast paths when one side
  /// is a boolean and the other side is a constant {0,1}.
  ///
  /// Cases handled:
  ///   - bool == 1  → assert(bool)
  ///   - 1 == bool  → assert(bool)
  ///   - bool == 0  → assert(!bool)
  ///   - 0 == bool  → assert(!bool)
  ///   - otherwise  → assert(lhs == rhs)
  ///
  /// Returns success after emitting IR.
  static LogicalResult
  emitAssertEqOptimized(OpBuilder &b, Location loc, Value lhsVal, Value rhsVal) {
    // --- Small helpers --------------------------------------------------------
    auto isBool = [](Value v) { return llvm::isa<pcl::BoolType>(v.getType()); };

    auto isConstOne = [](Value v) {
      if (auto ap = getPclConstAPInt(v)) {
        return ap->isOne();
      }
      return false;
    };
    auto isConstZero = [](Value v) {
      if (auto ap = getPclConstAPInt(v)) {
        return ap->isZero();
      }
      return false;
    };

    auto emitEqAssert = [&](Value l, Value r) {
      auto eq = b.create<pcl::CmpEqOp>(loc, l, r);
      b.create<pcl::AssertOp>(loc, eq.getRes());
    };

    auto emitAssertTrue = [&](Value pred) { b.create<pcl::AssertOp>(loc, pred); };

    auto emitAssertFalse = [&](Value pred) {
      auto neg = b.create<pcl::NotOp>(loc, pred);
      b.create<pcl::AssertOp>(loc, neg.getRes());
    };

    // Optimized handling of boolean patterns
    if (isBool(lhsVal) && isConstOne(rhsVal)) {
      // bool == 1 → assert(bool)
      emitAssertTrue(lhsVal);
      return success();
    }
    if (isBool(rhsVal) && isConstOne(lhsVal)) {
      // 1 == bool → assert(bool)
      emitAssertTrue(rhsVal);
      return success();
    }
    if (isBool(lhsVal) && isConstZero(rhsVal)) {
      // bool == 0 → assert(!bool)
      emitAssertFalse(lhsVal);
      return success();
    }
    if (isBool(rhsVal) && isConstZero(lhsVal)) {
      // 0 == bool → assert(!bool)
      emitAssertFalse(rhsVal);
      return success();
    }

    // Fallback to assert(lhs == rhs)
    emitEqAssert(lhsVal, rhsVal);
    return success();
  }

  /// Lower the constraint ops to PCL ops
  LogicalResult lowerStructToPCLBody(StructDefOp structDef, func::FuncOp dstFunc) {
    // Pull the field prime off the enclosing module
    auto primeAttr =
        dstFunc->getParentOfType<ModuleOp>()->getAttrOfType<pcl::PrimeAttr>("pcl.prime");
    assert(primeAttr && "pcl.prime attribute must be set before lowering");
    llvm::APInt prime = primeAttr.getValue().getValue();

    // As we build, map llzk values to their pcl ones
    llvm::DenseMap<Value, Value> llzkToPcl;
    OpBuilder b(dstFunc.getBody());
    // Map member name to PCL vars; public members are outputs, privates are intermediates
    llvm::DenseMap<StringRef, Value> member2pclvar;
    llvm::SmallVector<Value> outVars;

    // Create a new variable name for an `llzk.nondet` op with a paranoid check
    // that the generated name doesn't collide with any member names. The
    // counter is per-struct (PCL vars are function-scoped) so output does not
    // depend on what the process lowered earlier.
    auto getNondetVarName = [&member2pclvar, id = 0u]() mutable {
      std::string name;
      llvm::raw_string_ostream os(name);
      do {
        name.clear();
        os << "_nondet_internal_var__" << id;
        id++;
      } while (member2pclvar.contains(name));
      return name;
    };

    unsigned fullWidth = prime.getActiveBits();
    unsigned constBits = fullWidth + 1;

    auto feltConst = [&](Location loc, uint64_t v) -> Value {
      return b
          .create<pcl::ConstOp>(loc, pcl::FeltAttr::get(b.getContext(), llvm::APInt(constBits, v)))
          .getRes();
    };
    // Decomposition width of a constant.
    auto constWidth = [](const llvm::APInt &c) { return std::max(1u, c.getActiveBits()); };
    // Reads bit `i` of a decomposition, treating bits past its width as
    // constant zero; `zero` memoizes the constant across calls.
    auto bitAt = [&](Location loc, ArrayRef<Value> bits, unsigned i, Value &zero) -> Value {
      if (i < bits.size()) {
        return bits[i];
      }
      if (!zero) {
        zero = feltConst(loc, 0);
      }
      return zero;
    };

    // `cast.tofelt` forwards its operand's PCL value, so a felt LLZK value
    // can map to a `!pcl.bool` (e.g. a cmp result). PCL has no bool→felt
    // cast; materialize the {0,1} felt value as a witness `w` pinned by
    // `w·(w−1) == 0` and `b ⟺ (w == 1)`. Cached per bool value.
    llvm::DenseMap<Value, Value> boolFelt;
    auto asFelt = [&](Location loc, Value pclVal) -> Value {
      if (!llvm::isa<pcl::BoolType>(pclVal.getType())) {
        return pclVal;
      }
      auto [it, inserted] = boolFelt.try_emplace(pclVal);
      if (inserted) {
        Value w = b.create<pcl::VarOp>(loc, getNondetVarName(), /*is_output=*/false).getRes();
        Value zero = feltConst(loc, 0);
        Value one = feltConst(loc, 1);
        auto wMinus1 = b.create<pcl::SubOp>(loc, w, one);
        auto prod = b.create<pcl::MulOp>(loc, w, wMinus1.getRes());
        auto isZero = b.create<pcl::CmpEqOp>(loc, prod.getRes(), zero);
        b.create<pcl::AssertOp>(loc, isZero.getRes());
        auto eqOne = b.create<pcl::CmpEqOp>(loc, w, one);
        auto link = b.create<pcl::IffOp>(loc, pclVal, eqOne.getRes());
        b.create<pcl::AssertOp>(loc, link.getRes());
        it->second = w;
      }
      return it->second;
    };

    // Proven upper bounds (in bits) on values, from constants, range-guard
    // asserts, and propagation through lowered ops. Bounds must be implied by
    // already-emitted constraints or by op semantics, because a decomposition
    // at width `w` re-asserts `value < 2^w`; a wrong bound would reject honest
    // witnesses. Missing entry means full field width; the tightest bound wins.
    llvm::DenseMap<Value, unsigned> widthBound;
    auto setBound = [&](Value llzkVal, unsigned bits) {
      bits = std::max(1u, std::min(bits, fullWidth));
      if (bits == fullWidth) {
        return;
      }
      auto [it, inserted] = widthBound.try_emplace(llzkVal, bits);
      if (!inserted && bits < it->second) {
        it->second = bits;
      }
    };
    auto boundOf = [&](Value llzkVal) -> unsigned {
      if (auto it = llzkToPcl.find(llzkVal); it != llzkToPcl.end()) {
        if (auto c = getPclConstAPInt(it->second)) {
          return constWidth(*c);
        }
      }
      if (auto it = widthBound.find(llzkVal); it != widthBound.end()) {
        return it->second;
      }
      return fullWidth;
    };

    // Bits may only be cached (reused as a value's decomposition) when their
    // weighted sum is provably < p: any sum of fewer than fullWidth bits is
    // < 2^(fullWidth−1) < p. At full width the sum of mixed bits can exceed
    // p, in which case the bits describe a non-canonical representative of
    // the mod-p result.
    auto canCacheBits = [&](size_t numBits) { return numBits < fullWidth; };

    // Cache of checked bit decompositions, keyed by PCL value (low bit first).
    // Bitwise ops also record their results' bits (boolean by construction),
    // so chains of bitwise ops only decompose their leaves. Constants get a
    // literal 0/1 bit vector with no assertions. Vectors live in `bitsStorage`
    // so returned refs survive later cache insertions.
    std::deque<SmallVector<Value>> bitsStorage;
    llvm::DenseMap<Value, SmallVector<Value> *> bitsCache;
    auto getBits = [&](Location loc, Value llzkVal) -> ArrayRef<Value> {
      Value pclVal = llzkToPcl.lookup(llzkVal);
      assert(pclVal && "operand must be lowered before getBits");
      auto [it, inserted] = bitsCache.try_emplace(pclVal);
      if (inserted) {
        SmallVector<Value> &bits = bitsStorage.emplace_back();
        if (llvm::isa<pcl::BoolType>(pclVal.getType())) {
          // A bool's felt value is 0 or 1; its witness is already a checked
          // 1-bit decomposition.
          bits.push_back(asFelt(loc, pclVal));
        } else if (auto c = getPclConstAPInt(pclVal)) {
          for (unsigned i = 0, w = constWidth(*c); i < w; ++i) {
            bits.push_back(feltConst(loc, (*c)[i] ? 1 : 0));
          }
        } else {
          bits = decomposeBits(b, loc, pclVal, prime, getNondetVarName, boundOf(llzkVal));
        }
        it->second = &bits;
      }
      return *it->second;
    };
    auto cacheResultBits = [&](Value llzkResult, Value pclResult, SmallVector<Value> bits) {
      if (!canCacheBits(bits.size())) {
        return;
      }
      setBound(llzkResult, bits.size());
      bitsStorage.push_back(std::move(bits));
      bitsCache[pclResult] = &bitsStorage.back();
    };
    // Recomposes a slice of a checked decomposition as `llzkResult`'s value;
    // an empty slice is the constant 0. The slice becomes the result's bits.
    auto emitBitSlice = [&](Location loc, Value llzkResult, ArrayRef<Value> slice) {
      Value result;
      if (slice.empty()) {
        result = feltConst(loc, 0);
      } else {
        result = recomposeBits(b, loc, slice);
        cacheResultBits(llzkResult, result, SmallVector<Value>(slice));
      }
      rememberResult(llzkResult, result, llzkToPcl);
    };

    // Range guards: noir_llzk pins blackbox input widths by asserting
    // `cmp_lt(x, c) == 1` (with the bool cast to felt) before use; the guard
    // proves `x < c`, so later decompositions of `x` fit in bits(c−1) instead
    // of the full field width. Harvested over the whole block before lowering
    // because PCL constraints form one conjunction: a guard bounds `x`
    // wherever it appears, not just downstream of it.
    auto constAt = [](Value v) -> std::optional<llvm::APInt> {
      if (auto c = llvm::dyn_cast_if_present<FeltConstantOp>(v.getDefiningOp())) {
        return c.getValue().getValue();
      }
      if (auto c = llvm::dyn_cast_if_present<mlir::arith::ConstantOp>(v.getDefiningOp())) {
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(c.getValue())) {
          return intAttr.getValue();
        }
      }
      return std::nullopt;
    };
    auto harvestRangeGuard = [&](Value cmpSide, Value trueSide) {
      auto trueConst = constAt(trueSide);
      if (!trueConst || !trueConst->isOne()) {
        return;
      }
      // Look through `cast.tofelt`; noir_llzk casts the bool before
      // constraining it against the felt constant 1.
      if (auto castOp = llvm::dyn_cast_if_present<IntToFeltOp>(cmpSide.getDefiningOp())) {
        cmpSide = castOp.getValue();
      }
      auto cmp = llvm::dyn_cast_if_present<CmpOp>(cmpSide.getDefiningOp());
      if (!cmp) {
        return;
      }
      auto pred = cmp.getPredicate();
      Value bounded;
      std::optional<llvm::APInt> limit;
      if (pred == FeltCmpPredicate::LT || pred == FeltCmpPredicate::LE) {
        bounded = cmp.getLhs();
        limit = constAt(cmp.getRhs());
      } else if (pred == FeltCmpPredicate::GT || pred == FeltCmpPredicate::GE) {
        bounded = cmp.getRhs();
        limit = constAt(cmp.getLhs());
      } else {
        return;
      }
      if (!limit) {
        return;
      }
      bool strict = pred == FeltCmpPredicate::LT || pred == FeltCmpPredicate::GT;
      if (strict && limit->isZero()) {
        return;
      }
      llvm::APInt maxVal = strict ? *limit - 1 : *limit;
      setBound(bounded, maxVal.getActiveBits());
    };

    // Shared scaffolding for bitwise binary felt ops. Operands may have
    // different decomposition widths; the shorter side is padded with constant
    // zeros (sound: a width-`w` decomposition proves the value `< 2^w`). For
    // AND the result is truncated to the shorter width instead, since the
    // upper bits are all `x_i · 0`.
    auto lowerBitwiseBinary =
        [&](auto op, bool truncateToMin,
            llvm::function_ref<Value(Location, Value, Value)> mix) -> LogicalResult {
      auto lhs = lookup(op.getLhs(), llzkToPcl, op);
      if (failed(lhs)) {
        return failure();
      }
      auto rhs = lookup(op.getRhs(), llzkToPcl, op);
      if (failed(rhs)) {
        return failure();
      }
      auto loc = op.getLoc();
      ArrayRef<Value> lb = getBits(loc, op.getLhs());
      ArrayRef<Value> rb = getBits(loc, op.getRhs());
      unsigned n = truncateToMin ? std::min(lb.size(), rb.size()) : std::max(lb.size(), rb.size());
      Value zero;
      SmallVector<Value> ob;
      ob.reserve(n);
      for (unsigned i = 0; i < n; ++i) {
        ob.push_back(mix(loc, bitAt(loc, lb, i, zero), bitAt(loc, rb, i, zero)));
      }
      Value result = recomposeBits(b, loc, ob);
      rememberResult(op.getResult(), result, llzkToPcl);
      cacheResultBits(op.getResult(), result, std::move(ob));
      return success();
    };

    // Unary counterpart of `lowerBitwiseBinary`. Always mixes at full field
    // width (NOT complements every felt bit); bits above the operand's
    // decomposition width are constant zero.
    auto lowerBitwiseUnary = [&](auto op,
                                 llvm::function_ref<Value(Location, Value)> mix) -> LogicalResult {
      auto operand = lookup(op.getOperand(), llzkToPcl, op);
      if (failed(operand)) {
        return failure();
      }
      auto loc = op.getLoc();
      ArrayRef<Value> ab = getBits(loc, op.getOperand());
      Value zero;
      SmallVector<Value> ob;
      ob.reserve(fullWidth);
      for (unsigned i = 0; i < fullWidth; ++i) {
        ob.push_back(mix(loc, bitAt(loc, ab, i, zero)));
      }
      Value result = recomposeBits(b, loc, ob);
      rememberResult(op.getResult(), result, llzkToPcl);
      cacheResultBits(op.getResult(), result, std::move(ob));
      return success();
    };

    // Lowers `felt.shl(a, b) = a · 2^b mod p`.
    //
    // A constant `b = s` folds to a single multiplication by `2^s mod p`,
    // with the result's bits reused from `a`'s when they are cached and the
    // shifted value provably doesn't wrap.
    //
    // Otherwise only `b` is bit-decomposed. Let c_i = 2^(2^i) mod p
    // (precomputed at compile time by repeated squaring). Then
    //   2^b = ∏_i (b_i · (c_i − 1) + 1)  (mod p),
    // and the result is `a · 2^b`. Costs ~2 muls per bit plus one decomposition.
    auto lowerShl = [&](ShlFeltOp op) -> LogicalResult {
      auto lhs = lookup(op.getLhs(), llzkToPcl, op);
      auto rhs = lookup(op.getRhs(), llzkToPcl, op);
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      auto loc = op.getLoc();
      auto *ctx = b.getContext();

      if (auto shift = getPclConstAPInt(*rhs)) {
        llvm::APInt pow2s = toExactWidthAPInt(
            modExp(llvm::DynamicAPInt(2), toDynamicAPInt(*shift), toDynamicAPInt(prime)), constBits
        );
        auto pow2sConst = b.create<pcl::ConstOp>(loc, pcl::FeltAttr::get(ctx, pow2s));
        Value result = b.create<pcl::MulOp>(loc, asFelt(loc, *lhs), pow2sConst.getRes()).getRes();
        rememberResult(op.getResult(), result, llzkToPcl);
        unsigned s = shift->getLimitedValue(fullWidth);
        setBound(op.getResult(), boundOf(op.getLhs()) + s);
        // Reuse `a`'s cached bits shifted up by `s` when the product can't
        // wrap mod p, so downstream bitwise ops skip a decomposition.
        if (auto it = bitsCache.find(*lhs);
            it != bitsCache.end() && canCacheBits(it->second->size() + s)) {
          SmallVector<Value> shifted;
          shifted.reserve(it->second->size() + s);
          if (s > 0) {
            shifted.assign(s, feltConst(loc, 0));
          }
          shifted.append(it->second->begin(), it->second->end());
          cacheResultBits(op.getResult(), result, std::move(shifted));
        }
        return success();
      }

      unsigned wideBits = 2 * fullWidth + 4;
      llvm::APInt primeWide = prime.zext(wideBits);
      ArrayRef<Value> bBits = getBits(loc, op.getRhs());
      Value oneConst = feltConst(loc, 1);
      llvm::APInt curPow2 = llvm::APInt(wideBits, 2).urem(primeWide);

      Value pow2b = oneConst;
      for (unsigned i = 0, e = bBits.size(); i < e; ++i) {
        llvm::APInt cMinus1 = (curPow2 - llvm::APInt(wideBits, 1)).trunc(constBits);
        auto cm1Const = b.create<pcl::ConstOp>(loc, pcl::FeltAttr::get(ctx, cMinus1));
        auto scaled = b.create<pcl::MulOp>(loc, bBits[i], cm1Const.getRes());
        auto factor = b.create<pcl::AddOp>(loc, scaled.getRes(), oneConst);
        pow2b = b.create<pcl::MulOp>(loc, pow2b, factor.getRes()).getRes();
        curPow2 = (curPow2 * curPow2).urem(primeWide);
      }
      auto result = b.create<pcl::MulOp>(loc, asFelt(loc, *lhs), pow2b);
      rememberResult(op.getResult(), result.getRes(), llzkToPcl);
      return success();
    };

    // Lowers `felt.shr(a, b) = floor(a / 2^b)` on the unsigned integer
    // representative of `a`, treating shifts of `a`'s width or more as 0.
    //
    // A constant `b = s` is just a slice of `a`'s checked decomposition:
    // bits [s..w). Otherwise the low `L = ceil(log2(n))` bits of `b` drive a
    // barrel shifter over the bits of `a`; any higher bit of `b` being set
    // forces the result to 0 via a multiplicative gate.
    auto lowerShr = [&](ShrFeltOp op) -> LogicalResult {
      auto lhs = lookup(op.getLhs(), llzkToPcl, op);
      auto rhs = lookup(op.getRhs(), llzkToPcl, op);
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      auto loc = op.getLoc();

      if (auto shift = getPclConstAPInt(*rhs)) {
        ArrayRef<Value> aBits = getBits(loc, op.getLhs());
        unsigned s = shift->getLimitedValue(aBits.size());
        emitBitSlice(loc, op.getResult(), aBits.drop_front(s));
        return success();
      }

      unsigned L = (fullWidth > 1) ? llvm::APInt(32, fullWidth - 1).getActiveBits() : 0;

      ArrayRef<Value> aBits = getBits(loc, op.getLhs());
      ArrayRef<Value> bBits = getBits(loc, op.getRhs());

      Value zeroConst = feltConst(loc, 0);
      Value oneConst = feltConst(loc, 1);

      // Barrel shifter over `L` levels: at level i, conditionally shift right
      // by 2^i controlled by bBits[i]. Positions past the top become 0.
      // `a`'s bits are padded to full width with constant zeros; `b`'s bits
      // beyond its decomposition width are implicitly zero.
      SmallVector<Value> cur(aBits.begin(), aBits.end());
      cur.resize(fullWidth, zeroConst);
      for (unsigned i = 0; i < L && i < bBits.size(); ++i) {
        unsigned shift = 1u << i;
        Value ctrl = bBits[i];
        SmallVector<Value> next(fullWidth);
        for (unsigned j = 0; j < fullWidth; ++j) {
          Value neighbor = (j + shift < fullWidth) ? cur[j + shift] : zeroConst;
          auto diff = b.create<pcl::SubOp>(loc, neighbor, cur[j]);
          auto scaled = b.create<pcl::MulOp>(loc, ctrl, diff.getRes());
          next[j] = b.create<pcl::AddOp>(loc, cur[j], scaled.getRes()).getRes();
        }
        cur = std::move(next);
      }

      // If any bit of `b` at index >= L is set, the shift exceeds the value's
      // width, so gate every result bit to 0.
      Value inRange = oneConst;
      for (unsigned i = L, e = bBits.size(); i < e; ++i) {
        auto complement = b.create<pcl::SubOp>(loc, oneConst, bBits[i]);
        inRange = b.create<pcl::MulOp>(loc, inRange, complement.getRes()).getRes();
      }
      for (unsigned j = 0; j < fullWidth; ++j) {
        cur[j] = b.create<pcl::MulOp>(loc, inRange, cur[j]).getRes();
      }

      rememberResult(op.getResult(), recomposeBits(b, loc, cur), llzkToPcl);
      return success();
    };

    // `v · w == 1` pins `w` to the unique inverse and is unsatisfiable for
    // `v == 0`, matching the dialect's requirement that divisors be non-zero.
    auto emitInverse = [&](Location loc, Value v) -> Value {
      v = asFelt(loc, v);
      auto w = b.create<pcl::VarOp>(loc, getNondetVarName(), /*is_output=*/false);
      Value oneConst = feltConst(loc, 1);
      auto prod = b.create<pcl::MulOp>(loc, v, w.getRes());
      auto eq = b.create<pcl::CmpEqOp>(loc, prod.getRes(), oneConst);
      b.create<pcl::AssertOp>(loc, eq.getRes());
      return w.getRes();
    };

    // For a constant divisor c = 2^s, quotient and remainder are slices of
    // the dividend's checked bit decomposition: bits [s..w) and [0..s).
    // Dynamic and non-power-of-two divisors are rejected: a sound encoding of
    // `a == q·b + r` for dynamic `b` needs a multiprecision product argument
    // to rule out field wraparound (e.g. `b = p−1, a = 0` admits the forged
    // `q = 1, r = 1`), and nothing currently emitted into `@constrain` needs
    // it.
    auto lowerDivModPow2 = [&](auto op, bool wantQuotient) -> LogicalResult {
      auto lhs = lookup(op.getLhs(), llzkToPcl, op);
      auto rhs = lookup(op.getRhs(), llzkToPcl, op);
      if (failed(lhs) || failed(rhs)) {
        return failure();
      }
      auto divisor = getPclConstAPInt(*rhs);
      if (!divisor) {
        return op.emitError("PCL lowering only supports constant divisors");
      }
      if (!divisor->isPowerOf2()) {
        return op.emitError("PCL lowering only supports power-of-two divisors");
      }
      unsigned s = divisor->logBase2();
      auto loc = op.getLoc();
      ArrayRef<Value> bits = getBits(loc, op.getLhs());
      // A shift of the value's full width or more leaves no quotient bits.
      unsigned split = std::min<unsigned>(s, bits.size());
      emitBitSlice(
          loc, op.getResult(), wantQuotient ? bits.drop_front(split) : bits.take_front(split)
      );
      return success();
    };

    auto srcFunc = structDef.getConstrainFuncOp();
    auto srcArgs = srcFunc.getArguments().drop_front();
    auto dstArgs = dstFunc.getArguments();
    if (dstArgs.size() != srcArgs.size()) {
      return srcFunc.emitError("arg count mismatch after dropping self");
    }

    // 1-1 mapping of args from constraint args to PCL args
    for (auto [src, dst] : llvm::zip(srcArgs, dstArgs)) {
      llzkToPcl.try_emplace(src, dst);
    }
    for (auto memberDef : structDef.getMemberDefs()) {
      // Create a PCL var for each struct member. Public members are outputs in PCL
      auto pclVar =
          b.create<pcl::VarOp>(memberDef.getLoc(), memberDef.getName(), memberDef.hasPublicAttr());
      member2pclvar.insert({memberDef.getName(), pclVar});
      if (memberDef.hasPublicAttr()) {
        outVars.push_back(pclVar);
      }
    }
    if (!srcFunc.getBody().hasOneBlock()) {
      return srcFunc.emitError(
          "llzk-to-pcl translation assumes the constrain function body has 1 block"
      );
    }
    Block &srcEntry = srcFunc.getBody().front();
    for (Operation &op : srcEntry) {
      if (auto eq = llvm::dyn_cast<EmitEqualityOp>(&op)) {
        harvestRangeGuard(eq.getLhs(), eq.getRhs());
        harvestRangeGuard(eq.getRhs(), eq.getLhs());
      }
    }
    // Translate each op. Almost 1-1 and currently only support Felt/Bool ops.
    // TODO: Support calls, if-else, globals/lookups.
    for (Operation &op : srcEntry) {
      LogicalResult res = success();
      llvm::TypeSwitch<Operation *, void>(&op)
          .Case<FeltConstantOp>([&b, &llzkToPcl, &prime, &res](auto c) {
        res = lowerConst(b, c, prime, llzkToPcl);
      })
          .Case<mlir::arith::ConstantOp>([&b, &llzkToPcl, &prime, &res](auto c) {
        res = lowerConst(b, c, prime, llzkToPcl);
      })
          .Case<NonDetOp>([&b, &getNondetVarName, &llzkToPcl](auto n) {
        auto varName = getNondetVarName();
        auto pclVar = b.create<pcl::VarOp>(n.getLoc(), varName, /* public */ false);
        rememberResult(n.getResult(), pclVar, llzkToPcl);
      })
          .Case<AddFeltOp>([&](auto a) {
        res = lowerBinaryLike<AddFeltOp, pcl::AddOp>(b, a, llzkToPcl);
        if (succeeded(res)) {
          // No wrap while the bound stays below full width; the clamp in
          // setBound covers the rest (any canonical felt is < 2^fullWidth).
          setBound(a.getResult(), std::max(boundOf(a.getLhs()), boundOf(a.getRhs())) + 1);
        }
      })
          .Case<SubFeltOp>([&b, &llzkToPcl, &res](auto s) {
        res = lowerBinaryLike<SubFeltOp, pcl::SubOp>(b, s, llzkToPcl);
      })
          .Case<MulFeltOp>([&](auto m) {
        res = lowerBinaryLike<MulFeltOp, pcl::MulOp>(b, m, llzkToPcl);
        if (succeeded(res)) {
          setBound(m.getResult(), boundOf(m.getLhs()) + boundOf(m.getRhs()));
        }
      })
          .Case<NegFeltOp>([&b, &llzkToPcl, &res](auto n) {
        res = lowerUnaryLike<NegFeltOp, pcl::NegOp>(b, n, llzkToPcl);
      })
          .Case<AndFeltOp>([&b, &lowerBitwiseBinary, &res](AndFeltOp op) {
        // AND: out_i = a_i · b_i; bits past the shorter operand are all 0.
        res =
            lowerBitwiseBinary(op, /*truncateToMin=*/true, [&b](Location loc, Value av, Value bv) {
          return b.create<pcl::MulOp>(loc, av, bv).getRes();
        });
      })
          .Case<OrFeltOp>([&b, &lowerBitwiseBinary, &res](OrFeltOp op) {
        // OR: out_i = a_i + b_i − a_i · b_i
        res =
            lowerBitwiseBinary(op, /*truncateToMin=*/false, [&b](Location loc, Value av, Value bv) {
          auto sum = b.create<pcl::AddOp>(loc, av, bv);
          auto prod = b.create<pcl::MulOp>(loc, av, bv);
          return b.create<pcl::SubOp>(loc, sum.getRes(), prod.getRes()).getRes();
        });
      })
          .Case<XorFeltOp>([&b, &lowerBitwiseBinary, &res](XorFeltOp op) {
        // XOR: out_i = a_i + b_i − 2·a_i·b_i
        res =
            lowerBitwiseBinary(op, /*truncateToMin=*/false, [&b](Location loc, Value av, Value bv) {
          auto sum = b.create<pcl::AddOp>(loc, av, bv);
          auto prod = b.create<pcl::MulOp>(loc, av, bv);
          auto twoProd = b.create<pcl::AddOp>(loc, prod.getRes(), prod.getRes());
          return b.create<pcl::SubOp>(loc, sum.getRes(), twoProd.getRes()).getRes();
        });
      })
          .Case<NotFeltOp>([&b, &lowerBitwiseUnary, &res](NotFeltOp op) {
        // NOT: out_i = 1 − a_i (one's-complement of the decomposition).
        res = lowerBitwiseUnary(op, [&b](Location loc, Value av) {
          auto oneConst =
              b.create<pcl::ConstOp>(loc, pcl::FeltAttr::get(b.getContext(), llvm::APInt(2, 1)));
          return b.create<pcl::SubOp>(loc, oneConst.getRes(), av).getRes();
        });
      })
          .Case<ShlFeltOp>([&lowerShl, &res](ShlFeltOp op) { res = lowerShl(op); })
          .Case<ShrFeltOp>([&lowerShr, &res](ShrFeltOp op) { res = lowerShr(op); })
          .Case<UnsignedIntDivFeltOp>([&lowerDivModPow2, &res](UnsignedIntDivFeltOp op) {
        res = lowerDivModPow2(op, /*wantQuotient=*/true);
      })
          .Case<UnsignedModFeltOp>([&lowerDivModPow2, &res](UnsignedModFeltOp op) {
        res = lowerDivModPow2(op, /*wantQuotient=*/false);
      })
          .Case<InvFeltOp>([&llzkToPcl, &emitInverse, &res](InvFeltOp op) {
        auto operand = lookup(op.getOperand(), llzkToPcl, op);
        if (failed(operand)) {
          res = failure();
          return;
        }
        rememberResult(op.getResult(), emitInverse(op.getLoc(), *operand), llzkToPcl);
      })
          .Case<DivFeltOp>([&b, &llzkToPcl, &emitInverse, &asFelt, &res](DivFeltOp op) {
        // Not the cheaper `b · w == a` hint: that leaves `w` unconstrained
        // when a == b == 0. Inverting `b` keeps the result determined and
        // rejects b == 0.
        auto lhs = lookup(op.getLhs(), llzkToPcl, op);
        auto rhs = lookup(op.getRhs(), llzkToPcl, op);
        if (failed(lhs) || failed(rhs)) {
          res = failure();
          return;
        }
        auto loc = op.getLoc();
        Value invRhs = emitInverse(loc, *rhs);
        auto result = b.create<pcl::MulOp>(loc, asFelt(loc, *lhs), invRhs);
        rememberResult(op.getResult(), result.getRes(), llzkToPcl);
      })
          .Case<AndBoolOp>([&b, &llzkToPcl, &res](auto a) {
        res = lowerBinaryLike<AndBoolOp, pcl::AndOp>(b, a, llzkToPcl);
      })
          .Case<OrBoolOp>([&b, &llzkToPcl, &res](auto o) {
        res = lowerBinaryLike<OrBoolOp, pcl::OrOp>(b, o, llzkToPcl);
      })
          .Case<NotBoolOp>([&b, &llzkToPcl, &res](auto n) {
        res = lowerUnaryLike<NotBoolOp, pcl::NotOp>(b, n, llzkToPcl);
      })
          .Case<XorBoolOp>([&b, &llzkToPcl, &res](auto x) {
        // xor = ¬(a ⟺ b). Build the whole chain before mapping the result:
        // rememberResult never overwrites, so mapping the iff first would
        // permanently associate the result with the un-negated value.
        auto lhs = lookup(x.getLhs(), llzkToPcl, x);
        auto rhs = lookup(x.getRhs(), llzkToPcl, x);
        if (failed(lhs) || failed(rhs)) {
          res = failure();
          return;
        }
        auto loc = x.getLoc();
        auto iff = b.create<pcl::IffOp>(loc, *lhs, *rhs);
        auto not_op = b.create<pcl::NotOp>(loc, iff.getRes());
        rememberResult(x.getResult(), not_op.getResult(), llzkToPcl);
      })
          .Case<IntToFeltOp>([&llzkToPcl, &res](auto m) {
        auto arg = lookup(m.getValue(), llzkToPcl, m);
        if (failed(arg)) {
          res = failure();
          return;
        }
        rememberResult(m.getResult(), arg.value(), llzkToPcl);
      })
          .Case<CmpOp>([&b, &llzkToPcl, &res](auto cmp) {
        auto pred = cmp.getPredicate();
        switch (pred) {
        case FeltCmpPredicate::EQ:
          res = lowerBinaryLike<CmpOp, pcl::CmpEqOp>(b, cmp, llzkToPcl);
          break;
        case FeltCmpPredicate::NE: {
          // ne = ¬(a == b). Build the whole chain before mapping the result:
          // rememberResult never overwrites, so mapping the eq first would
          // permanently associate the result with the un-negated value.
          auto lhs = lookup(cmp.getLhs(), llzkToPcl, cmp);
          auto rhs = lookup(cmp.getRhs(), llzkToPcl, cmp);
          if (failed(lhs) || failed(rhs)) {
            res = failure();
            break;
          }
          auto loc = cmp.getLoc();
          auto eq = b.create<pcl::CmpEqOp>(loc, *lhs, *rhs);
          auto not_op = b.create<pcl::NotOp>(loc, eq.getRes());
          rememberResult(cmp.getResult(), not_op.getResult(), llzkToPcl);
          break;
        }
        case FeltCmpPredicate::LT:
          res = lowerBinaryLike<CmpOp, pcl::CmpLtOp>(b, cmp, llzkToPcl);
          break;
        case FeltCmpPredicate::LE:
          res = lowerBinaryLike<CmpOp, pcl::CmpLeOp>(b, cmp, llzkToPcl);
          break;
        case FeltCmpPredicate::GT:
          res = lowerBinaryLike<CmpOp, pcl::CmpGtOp>(b, cmp, llzkToPcl);
          break;
        case FeltCmpPredicate::GE:
          res = lowerBinaryLike<CmpOp, pcl::CmpGeOp>(b, cmp, llzkToPcl);
          break;
        }
      })
          .Case<EmitEqualityOp>([&](auto eq) {
        auto lhs = lookup(eq.getLhs(), llzkToPcl, eq);
        auto rhs = lookup(eq.getRhs(), llzkToPcl, eq);
        if (failed(lhs) || failed(rhs)) {
          res = failure();
          return;
        }

        auto loc = eq.getLoc();
        if (failed(emitAssertEqOptimized(b, loc, *lhs, *rhs))) {
          res = failure();
          return;
        }
      })
          .Case<MemberReadOp>([&member2pclvar, &llzkToPcl, &srcFunc](auto read) {
        // At this point every member in the struct should have a var associated with it
        // so we should simply retrieve the var associated with the member.
        (void)srcFunc; // to silence unused variable warning if asserts are disabled
        assert(read.getComponent() == srcFunc.getArguments()[0]);
        if (auto it = member2pclvar.find(read.getMemberName()); it != member2pclvar.end()) {
          rememberResult(read.getResult(), it->getSecond(), llzkToPcl);
        } else {
          llvm_unreachable("Every member should have been mapped to a pcl var");
        }
      })
          .Case<ReturnOp>([&b, &outVars](auto ret) {
        // We return all the output vars we defined above.
        b.create<pcl::ReturnOp>(
            ret.getLoc(), (llvm::SmallVector<Value>(outVars.begin(), outVars.end()))
        );
      }).Default([&res](Operation *unknown) {
        unknown->emitError("unsupported op in PCL lowering: ") << unknown->getName();
        res = failure();
      });
      if (failed(res)) {
        return failure();
      }
    }
    return success();
  }

  FailureOr<func::FuncOp> buildPCLFunc(StructDefOp structDef) {
    SmallVector<Type> pclInputTypes, pclOutputTypes;
    auto constrainFunc = structDef.getConstrainFuncOp();
    auto *ctx = structDef.getContext();
    for (auto arg : constrainFunc.getArguments().drop_front()) {
      auto argType = arg.getType();
      if (!llvm::isa<FeltType>(argType)) {
        return constrainFunc.emitError()
               << "Constrain function's args are expected to be felts. Found " << argType
               << "for arg #: " << arg.getArgNumber();
      }
      pclInputTypes.push_back(pcl::FeltType::get(ctx));
    }
    for (auto member : structDef.getMemberDefs()) {
      auto memberType = member.getType();
      if (!llvm::isa<FeltType>(memberType)) {
        return structDef.emitError() << "Member must be felt type. Found " << memberType
                                     << " for member: " << member.getName();
      }
      if (member.hasPublicAttr()) {
        pclOutputTypes.push_back(pcl::FeltType::get(ctx));
      }
    }
    FunctionType fty = FunctionType::get(ctx, pclInputTypes, pclOutputTypes);
    auto func = func::FuncOp::create(constrainFunc.getLoc(), structDef.getName(), fty);
    func.addEntryBlock();
    return func;
  }

  // PCL programs require a module-level attribute specifying the prime.
  LogicalResult setPrime(ModuleOp &newMod, ModuleOp &oldMod) {
    auto prime = selectPrime(oldMod);
    if (failed(prime)) {
      return failure();
    }

    // Add an extra bit to avoid the prime being represented as a negative number
    auto newBitWidth = prime->getBitWidth() + 1;
    auto ty = IntegerType::get(newMod.getContext(), newBitWidth);
    auto intAttr = IntegerAttr::get(ty, prime->zext(newBitWidth));
    newMod->setAttr("pcl.prime", pcl::PrimeAttr::get(newMod.getContext(), intAttr));

    return success();
  }

  FailureOr<llvm::APSInt> selectPrime(ModuleOp &module) {
    FieldSet fields;
    // If the collection reports that at least one FeltType did not declare the field and
    // the fields set is empty, then we raise an error.
    if (failed(collectFields(module, fields)) && fields.empty()) {
      return module->emitOpError() << "could not deduce the prime field";
    }
    // If the fields is empty and we reached this point it means that the IR we are about to lower
    // does not have a single felt type (because felts without a field will make `collectFields`
    // return failure). We return an error here since we don't have a prime to emit. In practice,
    // this situation it's going to be unlikely.
    if (fields.empty()) {
      return module->emitOpError()
             << "does not contain felt types and prime field couldn't be deduced";
    }
    // The pass only supports having one field for the whole circuit.
    if (fields.size() > 1) {
      return module->emitOpError() << "multiple fields is not supported";
    }
    const auto &selectedField = *(fields.begin());
    return toAPSInt(selectedField.get().prime());
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    // check PCLDialect is loaded.
    assert(moduleOp->getContext()->getLoadedDialect<pcl::PCLDialect>() && "PCL dialect not loaded");
    // Create the PCL module
    auto newMod = ModuleOp::create(moduleOp.getLoc());
    // Set the prime attribute
    if (failed(setPrime(newMod, /*oldMod=*/moduleOp))) {
      signalPassFailure();
      return;
    }
    // Convert each struct to a PCL function
    auto walkResult = moduleOp.walk([this, &newMod](StructDefOp structDef) -> WalkResult {
      // 1) verify the struct can be converted to PCL
      if (failed(validateStruct(structDef))) {
        return WalkResult::interrupt();
      }
      // 2) Construct the PCL function op but with an empty body
      FailureOr<func::FuncOp> pclFuncOp = buildPCLFunc(structDef);
      if (failed(pclFuncOp)) {
        return WalkResult::interrupt();
      }
      // 3) Fill in the PCL function body
      newMod.getBody()->push_back(*pclFuncOp);
      if (failed(lowerStructToPCLBody(structDef, pclFuncOp.value()))) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    // clear the original ops
    moduleOp.getRegion().takeBody(newMod.getBodyRegion());
    // Replace the module attributes
    moduleOp->setAttrs(newMod->getAttrDictionary());
    newMod.erase();
  }
};
} // namespace
