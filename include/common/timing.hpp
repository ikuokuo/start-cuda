#pragma once

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace times {

class Timing {
 public:
  Timing() = default;
  virtual ~Timing() = default;

  static std::shared_ptr<Timing> Create(std::string label = "Timing");
  virtual void Reset(std::string /*label*/) {}
  virtual void Reset() {}
  virtual void AddSplit(std::string /*split_label*/) {}
  virtual void DumpToLog(std::ostream &/*os*/ = std::cout) {}
};

class TimingImpl : public Timing {
 public:
  using clock = std::chrono::system_clock;

  explicit TimingImpl(std::string label) {
    Reset(std::move(label));
  }

  void Reset(std::string label) override {
    label_ = std::move(label);
    Reset();
  }

  void Reset() override {
    split_labels_.clear();
    split_times_.clear();
    AddSplit("");
  }

  void AddSplit(std::string split_label) override {
    split_labels_.push_back(std::move(split_label));
    split_times_.push_back(clock::now());
  }

  void DumpToLog(std::ostream &os) override {
    using namespace std::chrono;  // NOLINT
    os << label_ << ": begin" << std::endl;
    auto first = split_times_[0];
    auto now = first;
    for (std::size_t i = 1, n = split_times_.size(); i < n; i++) {
      now = split_times_[i];
      auto split_label = split_labels_[i];
      auto prev = split_times_[i - 1];
      os << label_ << ":      "
         << duration_cast<milliseconds>(now - prev).count() << " ms, "
         << split_label << std::endl;
    }
    os << label_ << ": end, "
       << duration_cast<milliseconds>(clock::now() - first).count() << " ms"
       << std::endl;
  }

 private:
  std::string label_;
  std::vector<std::string> split_labels_;
  std::vector<clock::time_point> split_times_;
};

inline
std::shared_ptr<Timing> Timing::Create(std::string label) {
  return std::make_shared<TimingImpl>(std::move(label));
}

}  // namespace times
