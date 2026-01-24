#pragma once

#include "defines.h"
#include "pch.h"

namespace nuri {

struct ValueTag {};
struct ErrorTag {};

template <typename R, typename E> class Result {
public:
  Result(ValueTag, const R &value) : hasValue_(true) {
    new (&storage_.value) R(value);
  }

  Result(ValueTag, R &&value) : hasValue_(true) {
    new (&storage_.value) R(std::move(value));
  }

  Result(ErrorTag, const E &error) : hasValue_(false) {
    new (&storage_.error) E(error);
  }

  Result(ErrorTag, E &&error) : hasValue_(false) {
    new (&storage_.error) E(std::move(error));
  }

  Result(const Result &other) : hasValue_(other.hasValue_) {
    if (hasValue_) {
      new (&storage_.value) R(other.storage_.value);
    } else {
      new (&storage_.error) E(other.storage_.error);
    }
  }

  Result(Result &&other) noexcept(std::is_nothrow_move_constructible_v<R> &&
                                  std::is_nothrow_move_constructible_v<E>)
      : hasValue_(other.hasValue_) {
    if (hasValue_) {
      new (&storage_.value) R(std::move(other.storage_.value));
    } else {
      new (&storage_.error) E(std::move(other.storage_.error));
    }
  }

  Result &operator=(const Result &other) {
    if (this != &other) {
      Result tmp(other);
      swap(tmp);
    }
    return *this;
  }

  Result &
  operator=(Result &&other) noexcept(std::is_nothrow_move_constructible_v<R> &&
                                     std::is_nothrow_move_constructible_v<E>) {
    if (this != &other) {
      destroy();
      hasValue_ = other.hasValue_;
      if (hasValue_) {
        new (&storage_.value) R(std::move(other.storage_.value));
      } else {
        new (&storage_.error) E(std::move(other.storage_.error));
      }
    }
    return *this;
  }

  ~Result() { destroy(); }

  [[nodiscard]] bool hasValue() const noexcept { return hasValue_; }
  [[nodiscard]] bool hasError() const noexcept { return !hasValue_; }
  [[nodiscard]] explicit operator bool() const noexcept { return hasValue_; }

  [[nodiscard]] R &value() & {
    if (!hasValue_) {
      throw std::runtime_error("Result does not contain a value");
    }
    return storage_.value;
  }

  [[nodiscard]] const R &value() const & {
    if (!hasValue_) {
      throw std::runtime_error("Result does not contain a value");
    }
    return storage_.value;
  }

  [[nodiscard]] R &&value() && {
    if (!hasValue_) {
      throw std::runtime_error("Result does not contain a value");
    }
    return std::move(storage_.value);
  }

  [[nodiscard]] E &error() & {
    if (hasValue_) {
      throw std::runtime_error("Result does not contain an error");
    }
    return storage_.error;
  }

  [[nodiscard]] const E &error() const & {
    if (hasValue_) {
      throw std::runtime_error("Result does not contain an error");
    }
    return storage_.error;
  }

  [[nodiscard]] E &&error() && {
    if (hasValue_) {
      throw std::runtime_error("Result does not contain an error");
    }
    return std::move(storage_.error);
  }

  [[nodiscard]] R &operator*() & { return value(); }
  [[nodiscard]] const R &operator*() const & { return value(); }
  [[nodiscard]] R &&operator*() && { return std::move(value()); }
  [[nodiscard]] R *operator->() { return &value(); }
  [[nodiscard]] const R *operator->() const { return &value(); }

  [[nodiscard]] static inline Result<R, E> makeResult(const R &value) {
    return Result<R, E>(ValueTag{}, value);
  }

  [[nodiscard]] static inline Result<R, E> makeResult(R &&value) {
    return Result<R, E>(ValueTag{}, std::forward<R>(value));
  }

  [[nodiscard]] static inline Result<R, E> makeError(const E &error) {
    return Result<R, E>(ErrorTag{}, error);
  }

  [[nodiscard]] static inline Result<R, E> makeError(E &&error) {
    return Result<R, E>(ErrorTag{}, std::forward<E>(error));
  }

  void swap(Result &rhs) noexcept {
    if (hasValue_ == rhs.hasValue_) {
      if (hasValue_) {
        R tmpValue(std::move(storage_.value));
        storage_.value.~R();
        new (&storage_.value) R(std::move(rhs.storage_.value));
        rhs.storage_.value.~R();
        new (&rhs.storage_.value) R(std::move(tmpValue));
      } else {
        E tmpError(std::move(storage_.error));
        storage_.error.~E();
        new (&storage_.error) E(std::move(rhs.storage_.error));
        rhs.storage_.error.~E();
        new (&rhs.storage_.error) E(std::move(tmpError));
      }
    } else {
      if (hasValue_) {
        E tmpError(std::move(rhs.storage_.error));
        rhs.storage_.error.~E();
        new (&rhs.storage_.value) R(std::move(storage_.value));
        storage_.value.~R();
        new (&storage_.error) E(std::move(tmpError));
      } else {
        R tmpValue(std::move(rhs.storage_.value));
        rhs.storage_.value.~R();
        new (&rhs.storage_.error) E(std::move(storage_.error));
        storage_.error.~E();
        new (&storage_.value) R(std::move(tmpValue));
      }
      std::swap(hasValue_, rhs.hasValue_);
    }
  }

private:
  void destroy() {
    if (hasValue_) {
      storage_.value.~R();
    } else {
      storage_.error.~E();
    }
  }

  union Storage {
    R value;
    E error;

    Storage() {}
    ~Storage() {}
  };

  Storage storage_;
  bool hasValue_;
};
} // namespace nuri