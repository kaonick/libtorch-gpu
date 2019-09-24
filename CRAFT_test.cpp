#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <string>

#include <chrono>

using namespace std;

torch::jit::script::Module module;
double text_threshold = 0.7;
double link_threshold = 0.4;
double low_text = 0.4;
bool poly = false;
bool isCUDA = false;//switch CUDA OR CPU

void loadModel(string& model_path) {
	cout << "path:" << model_path << "\n";


	torch::NoGradGuard no_grad_guard;
	if (isCUDA) {
		module = torch::jit::load(model_path, torch::kCUDA);
	}
	else {
		module = torch::jit::load(model_path, torch::kCPU);
	}

	//assert(module != nullptr);
	std::cout << "ok\n";
}

cv::Mat resize_aspect_ratio(cv::Mat& img, int square_size, int interpolation, double mag_ratio) {
	int h = img.rows;
	int w = img.cols;

	int target_size = mag_ratio * max(w, h);
	cout << "target_size=" << target_size << "\n";

	if (target_size > square_size) {
		target_size = square_size;
	}
	double ratio = (double)target_size / max(w, h);
	int target_w = (int)(w * ratio);
	int target_h = (int)(h * ratio);
	cout << "target_w=" << target_w << "\n";
	cout << "target_h=" << target_h << "\n";

	int target_w32 = target_w;
	int target_h32 = target_h;
	if (target_h % 32 != 0) {
		target_h32 = target_h + (32 - target_h % 32);
	}

	if (target_w % 32 != 0) {
		target_w32 = target_w + (32 - target_w % 32);
	}

	cv::Mat proc;
	cv::resize(img, proc, cv::Size(target_w32, target_h32), interpolation = interpolation);


	return proc;
}

vector<cv::Rect> getDetBoxes(torch::Tensor& score_text, torch::Tensor& score_link, double text_threshold, double link_threshold, double low_text, bool poly) {
	cout << "score_text sizes=" << score_text.sizes() << "\n";//[288, 384]
	cout << "score_text=" << score_text.size(0) << "," << score_text.size(1) << "\n";//score_text=288,384
	cout << "score_text dtype=" << score_text.dtype() << "\n";  //float

	cout << "long kfloat32=" << sizeof(torch::kFloat32) * score_text.numel() << "\n";//long=110592
	cout << "long float=" << sizeof(float) * score_text.numel() << "\n";//long=442368

	cv::Mat textmap(score_text.size(0), score_text.size(1), CV_32FC1);
	std::memcpy((void*)textmap.data, score_text.data_ptr(), sizeof(float) * score_text.numel());//sizeof(torch::kFloat32)

	cout << "textmap rows=" << textmap.rows << "\n"; //textmap rows = 288
	cout << "textmap=" << textmap.size() << "\n";//textmap = [384 x 288]
	cout << "textmap type=" << textmap.type() << "\n";

	cv::Mat linkmap(score_link.size(0), score_link.size(1), CV_32FC1);
	std::memcpy((void*)linkmap.data, score_link.data_ptr(), sizeof(float) * score_link.numel());

	cv::imshow("textmap", textmap);
	cv::imshow("linkmap", linkmap);

	int img_h = textmap.rows;
	int img_w = textmap.cols;

	//labeling method
	cv::Mat text_score, link_score;
	cv::threshold(textmap, text_score, low_text, 1, 0);
	cv::threshold(linkmap, link_score, link_threshold, 1, 0);

	//np(np1+np2)
	cv::Mat add_score;
	cv::add(text_score, link_score, add_score);
	cout << "add_score=" << add_score.size() << "\n";
	cout << "add_score type=" << add_score.type() << "\n";

	cv::Mat text_score_comb = cv::Mat(cv::Size(add_score.cols, add_score.rows), CV_32FC1);
	for (size_t row = 0; row < add_score.rows; row++) {
		float* ptr = add_score.ptr<float>(row);
		float* text_score_comb_ptr = text_score_comb.ptr<float>(row);
		for (size_t col = 0; col < add_score.cols; col++) {
			double value = ptr[col];

			if (value < 0) {
				text_score_comb_ptr[col] = 0;
			}
			else if (value > 1) {
				text_score_comb_ptr[col] = 1;
			}
			else {
				text_score_comb_ptr[col] = value;
			}
		}
	}
	cv::imshow("text_score_comb", text_score_comb);


	vector<cv::Rect> boundRect;
	text_score_comb.convertTo(text_score_comb, CV_8UC1);

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));//9
	cv::dilate(text_score_comb, text_score_comb, kernel);

	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierarchy;
	contours.clear();
	hierarchy.clear();
	findContours(text_score_comb, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<cv::Point> > contours_poly_new(contours.size());
	cout << "contours size=" << contours.size();
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(cv::Mat(contours[i]), contours_poly_new[i], 3, true);

		cv::Rect rect = cv::boundingRect(cv::Mat(contours_poly_new[i]));
		boundRect.push_back(rect);
	}

	return boundRect;
}




vector<cv::Rect> forword(cv::Mat& input_mat) {
	cv::Mat image_transfomed = resize_aspect_ratio(input_mat, 1320, cv::INTER_LINEAR, 1.5);
	image_transfomed.convertTo(image_transfomed, CV_32FC3);
	cv::cvtColor(image_transfomed, image_transfomed, cv::COLOR_BGR2RGB);

	cout << "input resize=" << image_transfomed.rows << "," << image_transfomed.cols << "\n";


	std::vector <float> mean_ = { 0.485f, 0.456f, 0.406f };
	std::vector <float> std_ = { 0.229f, 0.224f, 0.225f };
	for (size_t row = 0; row < image_transfomed.rows; row++) {
		cv::Vec3f* ptr = image_transfomed.ptr<cv::Vec3f>(row);
		for (size_t col = 0; col < image_transfomed.cols; col++) {
			for (int i = 0; i < 3; i++) {
				ptr[col][i] = (ptr[col][i] - (mean_[i] * 255.0)) / (std_[i] * 255.0);
			}
		}
	}
	cv::imshow("image_transfomed", image_transfomed);

	auto tensor_image = torch::from_blob(image_transfomed.data, { 1, image_transfomed.rows, image_transfomed.cols,image_transfomed.channels() }, torch::kFloat32);//kFloat32
	tensor_image = tensor_image.permute({ 0, 3, 1, 2 });
	cout << "before input\n";

	c10::IValue output;

	auto start = std::chrono::high_resolution_clock::now();
	if (isCUDA) {
		tensor_image = tensor_image.to(torch::kCUDA);
		output = module.forward({ tensor_image });
	}
	else {
		output = module.forward({ tensor_image });
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "*****forward waste Time(s) " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;

	torch::Tensor score_text, score_link;

	if (output.isTuple()) {
		cout << "output isTuple!" << '\n';
		auto ouputTuple = output.toTuple();

		cout << "tuple_size:" << ouputTuple.use_count() << '\n';
		torch::Tensor out_tensor = ouputTuple->elements()[0].toTensor();
		torch::Tensor out_tensor1 = ouputTuple->elements()[1].toTensor();
		cout << "out_tensor:" << out_tensor.sizes() << '\n';//out_tensor:[1, 288, 384, 2]  576,768
		cout << "out_tensor1:" << out_tensor1.sizes() << '\n';//out_tensor1:[1, 32, 288, 384]

		torch::Tensor score;
		if (isCUDA) {
			torch::Tensor new_out_tensor = out_tensor.to(torch::kCPU);
			score = new_out_tensor.squeeze(0);// [288, 384, 2]
		}
		else {
			score = out_tensor.squeeze(0);// [288, 384, 2]
		}

		score_text = score.slice(2, 0, 1, 1).squeeze(2); //[288, 384]
		score_link = score.slice(2, 1, 2, 1).squeeze(2);

		cout << "score:" << score.sizes() << '\n';
		cout << "score_text:" << score_text.sizes() << '\n';
		cout << "score_link:" << score_link.sizes() << '\n';

	}

	cout << "output_tensor:" << '\n';

	int h = score_text.size(0);
	int w = score_text.size(1);
	double radio_w = (double)input_mat.cols / w;
	double radio_h = (double)input_mat.rows / h;

	vector<cv::Rect> boxes = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly);
	for (int i = 0; i < boxes.size(); i++)
	{
		cv::rectangle(input_mat, cv::Rect(boxes[i].x * radio_w, boxes[i].y * radio_h, boxes[i].width * radio_w, boxes[i].height * radio_h), cv::Scalar(0, 255, 0), 2, 8, 0);
	}
	cv::imshow("input_mat", input_mat);

	return boxes;

}


int main(int argc, const char* argv[]) {
	at::init_num_threads();

	cout << "Choice 1:CUDA,2:CPU>\n";
	int type;
	cin >> type;
	cout << "You choice "<<type<<"\n";

	string model_path("E:/WS_Nick/WS_CPP/libtorch-gpu-20190924/craft_mlt_25k.pt");
	string image_path("E:/WS_Nick/WS_CPP/libtorch-gpu-20190924/images");

	if (type == 1) {
		isCUDA = true;
	}
	else {
		isCUDA = false;
	}

	loadModel(model_path);

	cv::Mat image = cv::imread(image_path +"/"+ "724982074_m.jpg",cv::ImreadModes::IMREAD_COLOR);

	cout <<"input raw size="<< image.rows << "," << image.cols << "\n";

	auto start = std::chrono::high_resolution_clock::now();

	vector<cv::Rect> output=forword(image);

	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "C++ Operation Time(s) " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;

	cout << "output size=" << output.size() << "\n";

    cv::imshow("image", image);
    cv::waitKey(0);


}
