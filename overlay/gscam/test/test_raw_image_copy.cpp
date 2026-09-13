#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gscam/raw_image_copy.h>

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

namespace {

TEST(RawImageCopy, PreservesTightlyPacked640By480Rgb) {
  const std::size_t row_bytes = 640u * 3u;
  const std::size_t height = 480u;
  std::vector<std::uint8_t> source(row_bytes * height);
  for (std::size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<std::uint8_t>(i % 251u);
  }

  std::vector<std::uint8_t> destination;
  std::string error;
  ASSERT_TRUE(gscam::copyRawImageRows(
      source.data(), source.size(), 0u, row_bytes, row_bytes, height,
      &destination, &error)) << error;
  EXPECT_EQ(source, destination);
}

TEST(RawImageCopy, RemovesRgbRowPaddingAndHonorsOffset) {
  const std::vector<std::uint8_t> source = {
      99u, 1u, 2u, 3u, 4u, 5u, 6u, 90u, 91u,
      7u,  8u, 9u, 10u, 11u, 12u, 92u, 93u};
  const std::vector<std::uint8_t> expected = {
      1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u};

  std::vector<std::uint8_t> destination;
  std::string error;
  ASSERT_TRUE(gscam::copyRawImageRows(
      source.data(), source.size(), 1u, 8u, 6u, 2u, &destination, &error))
      << error;
  EXPECT_EQ(expected, destination);
}

TEST(RawImageCopy, RemovesMonoRowPadding) {
  const std::vector<std::uint8_t> source = {
      1u, 2u, 3u, 200u, 4u, 5u, 6u, 201u};
  const std::vector<std::uint8_t> expected = {1u, 2u, 3u, 4u, 5u, 6u};

  std::vector<std::uint8_t> destination;
  std::string error;
  ASSERT_TRUE(gscam::copyRawImageRows(
      source.data(), source.size(), 0u, 4u, 3u, 2u, &destination, &error))
      << error;
  EXPECT_EQ(expected, destination);
}

TEST(RawImageCopy, RejectsShortStrideWithoutPublishingPartialData) {
  const std::vector<std::uint8_t> source(12u, 1u);
  std::vector<std::uint8_t> destination(4u, 9u);
  std::string error;
  EXPECT_FALSE(gscam::copyRawImageRows(
      source.data(), source.size(), 0u, 5u, 6u, 2u, &destination, &error));
  EXPECT_TRUE(destination.empty());
  EXPECT_FALSE(error.empty());
}

TEST(RawImageCopy, RejectsBufferUnderflow) {
  const std::vector<std::uint8_t> source(13u, 1u);
  std::vector<std::uint8_t> destination;
  std::string error;
  EXPECT_FALSE(gscam::copyRawImageRows(
      source.data(), source.size(), 0u, 8u, 6u, 2u, &destination, &error));
  EXPECT_TRUE(destination.empty());
}

TEST(RawImageCopy, RejectsArithmeticOverflow) {
  const std::vector<std::uint8_t> source(1u, 1u);
  std::vector<std::uint8_t> destination;
  std::string error;
  EXPECT_FALSE(gscam::copyRawImageRows(
      source.data(), source.size(), 0u,
      std::numeric_limits<std::size_t>::max(), 1u,
      std::numeric_limits<std::size_t>::max(), &destination, &error));
  EXPECT_TRUE(destination.empty());
}

}  // namespace
