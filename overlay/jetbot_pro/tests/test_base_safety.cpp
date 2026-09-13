#include <gtest/gtest.h>

#include <limits>

#include "jetbot_pro/base_safety.hpp"

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(BaseSafety, AcceptsAndClampsFiniteVelocity)
{
  jetbot_pro_safety::VelocityDecision accepted =
      jetbot_pro_safety::sanitizeVelocity(0.2, -0.1, 0.7, 0.7, 2.5);
  EXPECT_TRUE(accepted.input_finite);
  EXPECT_FALSE(accepted.clamped);
  EXPECT_DOUBLE_EQ(accepted.x, 0.2);

  jetbot_pro_safety::VelocityDecision clamped =
      jetbot_pro_safety::sanitizeVelocity(2.0, -3.0, 9.0, 0.7, 2.5);
  EXPECT_TRUE(clamped.input_finite);
  EXPECT_TRUE(clamped.clamped);
  EXPECT_DOUBLE_EQ(clamped.x, 0.7);
  EXPECT_DOUBLE_EQ(clamped.y, -0.7);
  EXPECT_DOUBLE_EQ(clamped.yaw, 2.5);
}

TEST(BaseSafety, RejectsEveryNonFiniteVelocity)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const double samples[3][3] = {{nan, 0.0, 0.0}, {0.0, inf, 0.0},
                                {0.0, 0.0, -inf}};
  for (int i = 0; i < 3; ++i)
  {
    jetbot_pro_safety::VelocityDecision result =
        jetbot_pro_safety::sanitizeVelocity(
            samples[i][0], samples[i][1], samples[i][2], 0.7, 2.5);
    EXPECT_FALSE(result.input_finite);
    EXPECT_DOUBLE_EQ(result.x, 0.0);
    EXPECT_DOUBLE_EQ(result.y, 0.0);
    EXPECT_DOUBLE_EQ(result.yaw, 0.0);
  }
}

TEST(BaseSafety, StopsForTimeoutClockResetAndSafetyLatch)
{
  EXPECT_FALSE(jetbot_pro_safety::shouldCommandStop(0.5, 1.0, true, false));
  EXPECT_FALSE(jetbot_pro_safety::shouldCommandStop(0.999, 1.0, true, false));
  EXPECT_TRUE(jetbot_pro_safety::shouldCommandStop(1.0, 1.0, false, false));
  EXPECT_TRUE(jetbot_pro_safety::shouldCommandStop(1.1, 1.0, false, false));
  EXPECT_TRUE(jetbot_pro_safety::shouldCommandStop(-0.1, 1.0, false, false));
  EXPECT_TRUE(jetbot_pro_safety::shouldCommandStop(
      std::numeric_limits<double>::quiet_NaN(), 1.0, false, false));
  EXPECT_TRUE(jetbot_pro_safety::shouldCommandStop(0.1, 1.0, true, true));
  EXPECT_FALSE(jetbot_pro_safety::shouldCommandStop(0.1, 1.0, false, true));
}

TEST(BaseSafety, CycleCommandZerosEveryAxisForTimeoutSafetyAndInvalidInput)
{
  jetbot_pro_safety::CycleCommand normal =
      jetbot_pro_safety::commandForCycle(
          0.2, -0.1, 0.7, 0.2, 1.0, true, false);
  EXPECT_FALSE(normal.stopped);
  EXPECT_DOUBLE_EQ(normal.x, 0.2);
  EXPECT_DOUBLE_EQ(normal.y, -0.1);
  EXPECT_DOUBLE_EQ(normal.yaw, 0.7);

  jetbot_pro_safety::CycleCommand timeout =
      jetbot_pro_safety::commandForCycle(
          0.2, -0.1, 0.7, 1.0, 1.0, false, false);
  EXPECT_TRUE(timeout.stopped);
  EXPECT_DOUBLE_EQ(timeout.x, 0.0);
  EXPECT_DOUBLE_EQ(timeout.y, 0.0);
  EXPECT_DOUBLE_EQ(timeout.yaw, 0.0);

  jetbot_pro_safety::CycleCommand safety =
      jetbot_pro_safety::commandForCycle(
          0.2, -0.1, 0.7, 0.1, 1.0, true, true);
  EXPECT_TRUE(safety.stopped);
  EXPECT_DOUBLE_EQ(safety.x, 0.0);
  EXPECT_DOUBLE_EQ(safety.y, 0.0);
  EXPECT_DOUBLE_EQ(safety.yaw, 0.0);

  jetbot_pro_safety::CycleCommand ignored_safety =
      jetbot_pro_safety::commandForCycle(
          0.2, -0.1, 0.7, 0.1, 1.0, false, true);
  EXPECT_FALSE(ignored_safety.stopped);

  jetbot_pro_safety::CycleCommand invalid =
      jetbot_pro_safety::commandForCycle(
          std::numeric_limits<double>::quiet_NaN(), 0.1, 0.2,
          0.1, 1.0, false, false);
  EXPECT_TRUE(invalid.stopped);
  EXPECT_DOUBLE_EQ(invalid.x, 0.0);
  EXPECT_DOUBLE_EQ(invalid.y, 0.0);
  EXPECT_DOUBLE_EQ(invalid.yaw, 0.0);
}

TEST(BaseSafety, AcceptsOnlyTheDocumentedBoundedTelemetryFrame)
{
  EXPECT_TRUE(jetbot_pro_safety::validTelemetryFrameSize(0x2D, 50, 0x2D));
  EXPECT_FALSE(jetbot_pro_safety::validTelemetryFrameSize(0, 50, 0x2D));
  EXPECT_FALSE(jetbot_pro_safety::validTelemetryFrameSize(3, 50, 3));
  EXPECT_FALSE(jetbot_pro_safety::validTelemetryFrameSize(0x2C, 50, 0x2D));
  EXPECT_FALSE(jetbot_pro_safety::validTelemetryFrameSize(51, 50, 51));
}

TEST(BaseSafety, RejectsInvalidOrOverflowingWireParameters)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(jetbot_pro_safety::validPositiveCorrection(1.0));
  EXPECT_FALSE(jetbot_pro_safety::validPositiveCorrection(0.0));
  EXPECT_FALSE(jetbot_pro_safety::validPositiveCorrection(nan));
  EXPECT_FALSE(jetbot_pro_safety::validPositiveCorrection(32.768));
  EXPECT_TRUE(jetbot_pro_safety::validVelocityLimit(0.0));
  EXPECT_TRUE(jetbot_pro_safety::validVelocityLimit(32.767));
  EXPECT_FALSE(jetbot_pro_safety::validVelocityLimit(-0.1));
  EXPECT_FALSE(jetbot_pro_safety::validVelocityLimit(inf));
  EXPECT_FALSE(jetbot_pro_safety::validVelocityLimit(32.768));
}
