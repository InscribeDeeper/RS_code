const showsRoutes = require("./shows");

const constructorMethod = (app) => {
	app.use("/", showsRoutes); // landing

	//http://localhost:3000/*
	app.use("*", (req, res) => {
		res.status(404).json({ error: "Not found url. Please edit url" });
	});
};

module.exports = constructorMethod;

