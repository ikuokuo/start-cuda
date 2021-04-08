#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#ifndef STR
  #define STR(x) #x
#endif
#ifndef STRINGIFY
  #define STRINGIFY(x) STR(x)
#endif

#ifndef MY_DATA_DIR
  #define MY_DATA_DIR STRINGIFY(MY_SAMPLES_DIR)"/data"
#endif
#ifndef MY_OUTPUT_DIR
  #define MY_OUTPUT_DIR STRINGIFY(MY_SAMPLES_DIR)
#endif

int main(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;

  auto img_path = MY_DATA_DIR"/lena.jpg";
  auto img = cv::imread(img_path, cv::IMREAD_COLOR);
  std::cout << "Read: " << img_path << std::endl;

  auto win_name = "lena";
  cv::namedWindow(win_name);

  cv::imshow(win_name, img);
  cv::waitKey();

  cv::destroyAllWindows();
  return 0;
}
