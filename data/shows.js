const axios = require("axios");

let exportedMethods = {
	async getShows() {
		const { data } = await axios.get("http://api.tvmaze.com/shows");
		const parsedData = data; // JSON.parse(JSON.stringify(data)) // parse the data from JSON into a normal JS Object
		return parsedData; // this will be the array of people objects
	},

	async getShowById(id) {
		if (id == null || !/^\d+$/.test(id)) throw "You must provide an numerical id to search for";
		if (id <= 0 || id % 1 !== 0) throw "You must provide an positive integer id to search for";

		// Retrieve all then check     // non-unique value match;
		// const Shows = await this.getShows(); // 如果需要打包, 必须要加个this, 如果没有外面打包, 则可以不加this
		// const show = Shows.filter((x) => x.id == Number.parseInt(id)); // should not use triple equal. otherwise, it will not match.
		// if (show.length == 0) throw "No Shows with that id";

		// Directly retrieve target based on route built on TVMAZE
		const show = await axios.get("http://api.tvmaze.com/shows/" + id);
		const showObj = show.data;
		if (showObj.hasOwnProperty("id") == false) throw "No Shows with that id";

		return showObj;
	},

	async showSearch(searchTerm) {
		// if (typeof(searchTerm) != "string") throw "You must provide an string";

		const { data } = await axios.get("http://api.tvmaze.com/search/shows?q=" + String(searchTerm));
		// the router from MAZE limit the output of ten rather then twenty!!!
		return data;
	},
};

module.exports = exportedMethods;

// exportedMethods
// 	.getShowById(11)
// 	.then((output) => console.log(output))
// 	.catch((error) => console.log(error));

	// exportedMethods
	// .showSearch('d')
	// .then((output) => console.log(output))
	// .catch((error) => console.log(error));

// console.log(typeof(searchTerm) != "string");
// console.log(!/^\d+$/.test("4215"));
// console.log(Number.parseInt("552") === 552);
