const rsRoutes = require("./tobedone");

const constructorMethod = (app) => {
	app.use("/", rsRoutes); // landing

	//http://localhost:3000/*
	app.use("*", (req, res) => {
		res.status(404).json({ error: "Not found url. Please edit url" });
	});
};

module.exports = constructorMethod;

