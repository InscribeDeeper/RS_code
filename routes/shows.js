const express = require("express");
const router = express.Router();
const data = require("../data");
const showData = data.shows;

// root route : localhost/
router.get("/", async (req, res) => {
	try {
		res.render("TV_maze/searchPage", { title: "Show Finder" }); // handlebar on views
	} catch (e) {
		res.status(500).json({ error: "No response from server" });
	}
});

// localhost/search
router.post("/search", async (req, res) => {
	let reqData = req.body.input;
	let errors = []; // 用list 来保存 errors, 最后来一起返回

	let maxOutput = 20; // the router from MAZE limit the output of ten rather then twenty!!!

	if (!reqData || reqData == null || reqData.trim().length == 0) {
		errors.push("No search terms provided or Only space provided");
	}

	if (errors.length > 0) {
		res.render("TV_maze/searchPage", {
			errors: errors,
			hasErrors: true,
			searchTerm: reqData,
			title: "Show Finder",
			// 这里有解决重新输入的问题, 因为把 req.body的数据重新用post这个key传了进去. 然后在render的界面上, 会附带额外的内容
		});
		return;
	}

	try {
		const outputList = await showData.showSearch(reqData);

		if (outputList.length == 0) {
			res.render("TV_maze/searchOutputList", {
				notFound: true,
				Found: false,
				searchTerm: reqData,
				title: "Shows not Found",
			});
		} else {
			res.render("TV_maze/searchOutputList", {
				notFound: false,
				Found: true,
				shows: outputList.slice(0, maxOutput),
				searchTerm: reqData,
				title: "Shows Found",
			});
		}
	} catch (e) {
		res.status(404).json({ error: e });
	}
});

router.get("/shows/:id", async (req, res) => {
	const maxIdNumber = 54669;
	const singleShow = await showData.getShowById(req.params.id);
	
	if (req.body > maxIdNumber) {
		res.status(404).json({ error: "index out of boundary 54669" });
	};
	
	res.render("TV_maze/showDetails", { show: singleShow, parsedSummary: singleShow.summary.replace(/<[^>]+>/ig, '')});
});

module.exports = router;
