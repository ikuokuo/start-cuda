#pragma once

#include <cmath>
#include <chrono>
#include <functional>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#include <unordered_map>

#include "times.hpp"

#ifndef TIME_PRECISION
  #define TIME_PRECISION 3
#endif

namespace times {

class TimeCost {
 public:
  using tag_map_t = std::unordered_map<std::string, std::shared_ptr<TimeCost>>;

  explicit TimeCost(const std::string &tag) : tag_(tag) {}
  ~TimeCost() {}

  std::ostream &ToString(std::ostream &os) const {
    float ms = times::count<std::chrono::microseconds>(elapsed()) * 0.001f;
    os << tag_ << std::endl
      << "BEG: " << times::to_local_string(beg_, "%F %T", TIME_PRECISION)
      << std::endl
      << "END: " << times::to_local_string(end_, "%F %T", TIME_PRECISION)
      << std::endl
      << "COST: " << ms << " ms";
    return os;
  }

  std::ostream &ToLineString(std::ostream &os) const {
    float ms = times::count<std::chrono::microseconds>(elapsed()) * 0.001f;
    os << tag_ << ": " << ms << " ms, "
      << times::to_local_string(beg_, "%T", TIME_PRECISION) << " > "
      << times::to_local_string(end_, "%T", TIME_PRECISION);
    return os;
  }

  std::string ToString() const {
    std::stringstream ss;
    ToString(ss);
    return ss.str();
  }

  std::string ToLineString() const {
    std::stringstream ss;
    ToLineString(ss);
    return ss.str();
  }

  std::string tag() const { return tag_; }
  times::clock::time_point beg() const { return beg_; }
  times::clock::time_point end() const { return end_; }
  times::clock::duration elapsed() const { return end_ - beg_; }

  void set_beg(const times::clock::time_point &t) { beg_ = t; }
  void set_end(const times::clock::time_point &t) { end_ = t; }

  static std::shared_ptr<TimeCost> Beg(const std::string &tag) {
    const std::lock_guard<std::mutex> lock(GetMutex());
    std::shared_ptr<TimeCost> cost = std::make_shared<TimeCost>(tag);
    cost->beg_ = times::now();
    tag_map_t &map = GetTagMap();
    auto it = map.insert({tag, cost});
    if (!it.second) {
      throw std::logic_error("This tag already in use");
    }
    return it.first->second;
  }

  static std::shared_ptr<TimeCost> End(const std::string &tag) {
    const std::lock_guard<std::mutex> lock(GetMutex());
    tag_map_t &map = GetTagMap();
    auto it = map.find(tag);
    if (it == map.end()) {
      throw std::logic_error("This tag not Beg before End");
    }
    std::shared_ptr<TimeCost> cost = it->second;
    cost->end_ = times::now();
    map.erase(it);
    return cost;
  }

 private:
  std::string tag_;
  times::clock::time_point beg_;
  times::clock::time_point end_;

  static tag_map_t  &GetTagMap() { static tag_map_t map;  return map; }  // NOLINT
  static std::mutex &GetMutex()  { static std::mutex mtx; return mtx; }  // NOLINT
};

}  // namespace times

#ifdef TIME_COST
  #define TIME_BEG(tag) times::TimeCost::Beg(tag)
  #define TIME_END(tag) (times::TimeCost::End(tag)->ToLineString(std::cout) \
    << std::endl)
  #define TIME_BEG_FUNC(tag) do { \
    std::stringstream ss; \
    ss << __func__ << "::" << tag; \
    TIME_BEG(ss.str()); \
  } while (0)
  #define TIME_END_FUNC(tag) do { \
    std::stringstream ss; \
    ss << __func__ << "::" << tag; \
    TIME_END(ss.str()); \
  } while (0)
  #define TIME_BEG_FUNC2 TIME_BEG(__func__)
  #define TIME_END_FUNC2 TIME_END(__func__)
#else
  #define TIME_BEG(tag)
  #define TIME_END(tag)
  #define TIME_BEG_FUNC(tag)
  #define TIME_END_FUNC(tag)
  #define TIME_BEG_FUNC2
  #define TIME_END_FUNC2
#endif
