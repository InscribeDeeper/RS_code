const dbConnection = require("../config/mongoConnection");
const data = require("../data/");
const userData = data.users;
const furnitureData = data.furniture;
const rentalData = data.rental;
const commentData = data.comments;
const toggleFn = data.toggleFn;
var exec = require("child_process").exec;

async function main() {
	var arg1 = "hello";
	var arg2 = "world";
	var filename = "data/server_py/wordcloud.py"; // 默认在当前命令行

	// 最好让 python 输出的内容是 json format
	// exec 相当于用命令行执行东西, 返回的东西直接输出到这里.
	// 而且对应的python环境, 也会在命令行 已经声明

	exec("python" + " " + filename + " " + arg1 + " " + arg2, function (err, stdout, stderr) {
		if (err) {
			console.log("stderr", err);
		}
		if (stdout) {
			console.log("stdout", stdout);
		}
	});
}

main().catch(console.log);



// var exec = require('child_process').exec;
// filename = 'test_pyjson.py'
// var editor={
//     "name":'alex',
//     "age":18,
//     "address":'fghjh'
// };
// var str = '{\\"name\\":\\"alex\\",\\"age\\":18,\\"address\\":\\"sdsdd\\"}'; //change the javascriptobject to jsonstring
// exec('python '+filename+' '+str,function(err,stdout,stdin){

//     if(err){
//         console.log('err');
//     }
//     if(stdout)
//     {
//         //parse the string
//         console.log(stdout);
//         var astr = stdout.split('\r\n').join('');//delete the \r\n
//         var obj = JSON.parse(astr);

//         console.log('name',obj.name);
//         console.log('age',obj.age);
//         console.log('gender',obj.gender);

//     }


// });