#include <iostream>
#include <fstream> 
#include <string.h>
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

// split line into vector of strings
vector<string> split_line(const string&, const string&);
// convert a sequence of string to n-gram, then write n-gram into csv file
vector<vector<string>> to_n_gram(vector<string>&, int n, ofstream& outfile);
// write n-gram into file
void write_into_file(ofstream&, vector<string>);


// Input: window_size, input_filename, outfilename,  *featureNames
int main(int argc, char* argv[]) {
	ifstream infile; // infile object used to read data
	ofstream outfile; // final output file
	string line; // each line read from input file
	unordered_map<string, int> dict; // define a map, mapping column name into column index
	vector<string> desired_attr; // the features that we want to use to create n-gram; for example, sku_Id, 
	vector<string> sequence_data; // a sentence of item
	int window_size = stoi(argv[1]); // windows size, a windows size of 2 indicates a 5-gram
	//cout << "please specify window size > "; cin >> window_size;
	string sortColumnName = argv[2];
	string filename = argv[3];
	string outfilename = argv[4];
	// receive argument from command line
	for (int i = 5; argv[i] != NULL; ++i) {
		desired_attr.push_back(argv[i]); cout << argv[i] << ", ";
	} cout << endl;
	// open input file, open output file
	infile.open(filename);
	outfile.open(outfilename);

	// read the first line: the header line
	// split the header line
	// define column name to column index map
	getline(infile, line);	
	vector<string> header = split_line(line, ",");
	for (int i = 0; i < header.size(); i++){
		dict[header[i]] = i; cout << header[i] << ", ";
	} cout << endl;
	
	
	// loop over all the line in input file
	// if the userId this line is equal to the userId of the previous line, 
	// then add line into sequence
	// if it's not equal, convert sequence into n-gram, then write into output file

	string userId = ""; // initiate userId as null
	int userIdCol = dict[sortColumnName]; // use name2index map to get index of user_Id column
	int j = 0;
	string previous_data = ""; // to prevent repetitious item
	while (getline(infile, line)) {
		vector<string> data_line = split_line(line, ",");
		//cout << line << endl; 
		if (data_line[userIdCol] != userId){
			userId = data_line[userIdCol];
			if (j == 0) j+=1; else to_n_gram(sequence_data, window_size, outfile);
			sequence_data.clear();
			//continue;
		}
		j ++;
		if ((j % 100000) == 0){
			 cout << line << endl;
		}
		string data = "";
		for (string attr: desired_attr){
			int col = dict[attr];
			data += data_line[col];
			data += "__";
		}
		if (previous_data != data){
			sequence_data.push_back(data);// for (string data: sequence_data) cout << data << ","; cout << endl;
			previous_data = data;
		}
	} 

	
	outfile.close();
	infile.close();
	return 0;
}

// split line into vector of string
vector<string> split_line(const string& line, const string& delimeter){
	vector<string> output;
	int left = 0, span = 0;
	for (int i = 0; i < line.size(); ++i){
		if (line[i] == delimeter[0]){
			if (span == 0){
				output.push_back("");
			}
			else{
				output.push_back(line.substr(left, span));
			}
			left = i + 1;
			span = 0;
		}
		else{
			span ++;
		}

	}
	if (span != 0 ) output.push_back(line.substr(left, span)); else output.push_back("");
	return output;
}


vector<vector<string>> to_n_gram(vector<string>& sequence, int n, ofstream& outfile){
	vector<vector<string>> output;

	for (int i = 0; i < sequence.size(); ++i){
		vector<string> n_gram;
		for (int j = -n; j <= n; ++j){
			if (((i + j) < 0) || (i + j) >= sequence.size()){
				n_gram.push_back("NONE");
			}
			else{
				string current_gram = sequence[i+j];
				n_gram.push_back(current_gram);
			}
		}
		output.push_back(n_gram);
		write_into_file(outfile, n_gram);
	}

	return output;
}


void write_into_file(ofstream& outfile, vector<string> n_gram){
	for (int i = 0; i < n_gram.size()-1; ++i){
		outfile << n_gram[i] << ",";
	}
	outfile << n_gram[n_gram.size()-1] << endl;
}