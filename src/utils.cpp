
#include "utils.h"
 #include <fcntl.h>
 #include <sys/types.h>
 #include <sys/stat.h>


// Function to decode a Base64 string to a binary data vector
static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";


static inline bool is_base64(unsigned char c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

// Function to decode Base64 string to binary data vector
std::string base64Decode(std::string const& encoded_string) {
  int in_len = encoded_string.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  unsigned char char_array_4[4], char_array_3[3];
  std::string ret;

  while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
    char_array_4[i++] = encoded_string[in_]; in_++;
    if (i ==4) {
      for (i = 0; i <4; i++)
        char_array_4[i] = base64_chars.find(char_array_4[i]);

      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (i = 0; (i < 3); i++)
        ret += char_array_3[i];
      i = 0;
    }
  }

  if (i) {
    for (j = i; j <4; j++)
      char_array_4[j] = 0;

    for (j = 0; j <4; j++)
      char_array_4[j] = base64_chars.find(char_array_4[j]);

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
  }

  return ret;
}


// Function to convert a Base64 string to cv::Mat image
cv::Mat ccustomutils::base64ToMat(const std::string base64_data) {
    // Decode Base64 to binary data
    std::string img_data = base64Decode(base64_data);

    std::vector<uchar> vectordata(img_data.begin(),img_data.end());

    // Convert binary data to cv::Mat using imdecode
    cv::Mat img = cv::imdecode(vectordata, cv::IMREAD_UNCHANGED);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    if (img.empty()) {
        std::cerr << "Could not decode image" << std::endl;
    }
    return img;
}


bool ccustomutils::doesFileExist(const std::string& filepath) {
    struct stat buffer;
    return (stat (filepath.c_str(), &buffer) == 0);
}


float* ccustomutils::blobFromImage(Mat& img) {
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    float* blob = new float[img.total() * 3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                blob[c * img_w * img_h + h * img_w + w] = (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return blob;
}


std::string ccustomutils::millisecondsToDateTimeString() {
    // Get the current time in milliseconds since the epoch
    auto now = std::chrono::system_clock::now();
    auto milliseconds_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    // Convert milliseconds since epoch to time_t (seconds since epoch)
    std::time_t time_in_seconds = milliseconds_since_epoch / 1000;

    // Convert to a readable format
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_in_seconds), "%Y-%m-%d %H:%M:%S");

    // Append milliseconds part
    oss << "." << std::setw(3) << std::setfill('0') << (milliseconds_since_epoch % 1000);

    return oss.str();
}