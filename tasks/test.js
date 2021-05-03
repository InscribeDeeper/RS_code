
// didnt update test.js yet


// const dbConnection = require('./config/mongoConnection');
// const movies = require('./data/movies');
// const { ObjectId } = require('mongodb');


// const main = async () => {

//     const db = await dbConnection();
//     await db.dropDatabase(); // clean first

//     const first = await movies.create("Bill and Ted Face the Music",
//                                             "Once told they'd save the universe during a time-traveling adventure, 2 would-be rockers from San Dimas, California find themselves as middle-aged dads still trying to crank out a hit song and fulfill their destiny.",
//                                             "PG-13", 
//                                             "1hr 31min",
//                                             "Comedy",
//                                             ["Keanu Reeves","Alex Winter"],
//                                             {director: "Dean Parisot", yearReleased: 2020}
//                                             );
//     console.log('==== Log the first created movie ====');
//     console.log(first);


//     const second = await movies.create("The Dark Knight",
//                                             "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
//                                             "R",
//                                             "2hr 32min",
//                                             "Action",
//                                             ["Christian Bale","Heath Ledger"],
//                                             {director: "Christopher Nolan", yearReleased: 2008})
//     console.log('==== Query all movies ====');
//     const allMovies1 = await movies.getAll();
//     console.log(allMovies1);



//     const third = await movies.create("Hidden Figures",
//                                             "The story of a team of female African-American mathematicians who served a vital role in NASA during the early years of the U.S. space program.",
//                                             "PG",
//                                             "2hr 7min",
//                                             "Drama",
//                                             ["Taraji P. Henson","Octavia Spencer", "Janelle Monáe"],
//                                             {director: "Theodore Melfi", yearReleased: 2016})
//     console.log('==== Log the third created movie ====');
//     console.log(third);
    


//     console.log('==== rename the first movie ====');
//     const updatedFirst = await movies.rename(first._id, "The first movie's title has been updated" )
//     console.log(updatedFirst);
    

//     console.log('==== remove the second movie ====');
//     const removedInfo = await movies.remove(second._id)
//     console.log(removedInfo);
    
    
//     console.log('==== Query all movies ====');
//     const allMovies2 = await movies.getAll();
//     console.log(allMovies2);



//     console.log('Task Done!');





//     console.log('Error checking ... ');

//     let newObjId = ObjectId(); //creates a new object ID
//     let nonExistID = newObjId.toString(); // converts the Object ID to string
    

//     console.log('test1');
//     await movies.create("The Dark Knight",
//                         "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
//                         "R",
//                         "2hr 32min",
//                         "Action",
//                         [],
//                         {director: "Christopher Nolan", yearReleased: 2008})
//     .then(output => console.log((output)))
//     .catch(error => console.log((error)));

   
//     console.log('test2');
//     await movies.create("The Dark Knight",
//                         "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
//                         "R",
//                         "2hr 32min",
//                         "Action",
//                         ["Christian Bale","Heath Ledger"],
//                         {director: "Christopher Nolan"})
//     .then(output => console.log((output)))
//     .catch(error => console.log((error)));


//     console.log('test3');
//     await movies.create("The Dark Knight",
//                         "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
//                         "R",
//                         "2hr 32min",
//                         "Action",
//                         ["Christian Bale","Heath Ledger"],
//                         {director: "Christopher Nolan", yearReleased: 1900})
//     .then(output => console.log((output)))
//     .catch(error => console.log((error)));


//     console.log('test4');
//     await movies.create("The Dark Knight",
//                         "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
//                         "R",
//                         "2hr 32min",
//                         "Action",
//                         ["Christian Bale","Heath Ledger"],
//                         {director: "Christopher Nolan", yearReleased: "2008"})
//     .then(output => console.log((output)))
//     .catch(error => console.log((error)));





//     console.log('test5');
//     await movies.remove(1568944)
//     .then(output => console.log((output)))
//     .catch(error => console.log((error)));

//     console.log('test6');
//     await movies.remove()
//     .then(output => console.log((output)))
//     .catch(error => console.log((error)));

//     console.log('test7');
//     await movies.remove(nonExistID)
//     .then(output => console.log((output)))
//     .catch(error => console.log((error)));




//     console.log('test8');
//     await movies.rename()
//     .then(output => console.log((output)))
//     .catch(error => console.log((error)));

//     console.log('test9');
//     await movies.rename(third._id, 123)
//     .then(output => console.log((output)))
//     .catch(error => console.log((error)));

//     console.log('test10');
//     await movies.rename(nonExistID, 'AAA')
//     .then(output => console.log((output)))
//     .catch(error => console.log((error)));


//     console.log('test11');
//     await movies.rename('Not_ObjectID_string', 'AAA')
//     .then(output => console.log((output)))
//     .catch(error => console.log((error)));


//     await db.serverConfig.close();
//     console.log('Error checking Done!');
//  }


// main().catch((error) => {
//   console.log(error);
// });
