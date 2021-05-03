// no changes
const express = require("express");
const app = express();
const static = express.static(__dirname + "/public");
const configRoutes = require("./routes");
const exphbs = require("express-handlebars"); // 这个是单独的包, 用来处理 templating

// setup for route
app.use("/public", static);
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// setup for template engine -- handlebar
app.engine("handlebars", exphbs({ defaultLayout: "main" }));
app.set("view engine", "handlebars");

configRoutes(app);

app.listen(3000, () => {
	console.log("We've now got a server!");
	console.log("Your routes will be running on http://localhost:3000");
});
