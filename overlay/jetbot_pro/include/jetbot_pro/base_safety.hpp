#ifndef JETBOT_PRO_BASE_SAFETY_HPP
#define JETBOT_PRO_BASE_SAFETY_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace jetbot_pro_safety
{

struct VelocityDecision
{
  double x;
  double y;
  double yaw;
  bool input_finite;
  bool clamped;
};

struct CycleCommand
{
  double x;
  double y;
  double yaw;
  bool stopped;
};

inline VelocityDecision sanitizeVelocity(double x,
                                         double y,
                                         double yaw,
                                         double max_linear,
                                         double max_angular)
{
  VelocityDecision result = {0.0, 0.0, 0.0, false, false};
  result.input_finite = std::isfinite(x) && std::isfinite(y) &&
                        std::isfinite(yaw);
  if (!result.input_finite)
    return result;

  result.x = std::max(-max_linear, std::min(max_linear, x));
  result.y = std::max(-max_linear, std::min(max_linear, y));
  result.yaw = std::max(-max_angular, std::min(max_angular, yaw));
  result.clamped = result.x != x || result.y != y || result.yaw != yaw;
  return result;
}

inline bool shouldCommandStop(double command_age_s,
                              double timeout_s,
                              bool require_safety_stop,
                              bool safety_stop_active)
{
  return !std::isfinite(command_age_s) || command_age_s < 0.0 ||
         command_age_s >= timeout_s ||
         (require_safety_stop && safety_stop_active);
}

inline CycleCommand commandForCycle(double x,
                                    double y,
                                    double yaw,
                                    double command_age_s,
                                    double timeout_s,
                                    bool require_safety_stop,
                                    bool safety_stop_active)
{
  CycleCommand result = {x, y, yaw, false};
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(yaw) ||
      shouldCommandStop(command_age_s, timeout_s,
                        require_safety_stop, safety_stop_active))
  {
    result.x = result.y = result.yaw = 0.0;
    result.stopped = true;
  }
  return result;
}

inline bool validTelemetryFrameSize(std::size_t frame_size,
                                    std::size_t buffer_size,
                                    std::size_t expected_size)
{
  return frame_size == expected_size && frame_size <= buffer_size &&
         frame_size >= 4;
}

inline bool fitsSignedMilliscale(double value)
{
  // The wire protocol stores value * 1000 in a signed 16-bit field.
  return std::isfinite(value) && value >= -32.768 && value <= 32.767;
}

inline bool validPositiveCorrection(double value)
{
  return value > 0.0 && fitsSignedMilliscale(value);
}

inline bool validVelocityLimit(double value)
{
  return value >= 0.0 && fitsSignedMilliscale(value);
}

}  // namespace jetbot_pro_safety

#endif  // JETBOT_PRO_BASE_SAFETY_HPP
