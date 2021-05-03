const dbConnection = require('../config/mongoConnection');
const data = require('../data/');
const bookData = data.books; 
const reviewData = data.reviews; 
const mongoCollections = require('../config/mongoCollections');
const books = mongoCollections.books;
const reviews = mongoCollections.reviews;



async function main() {
    const db = await dbConnection();
    await db.dropDatabase();

    const book_1 = await bookData.createBook( 
                                            "The Shining",
                                            "Stephen", 
                                            "King",
                                            ["Novel", "Horror fiction", "Gothic fiction", "Psychological horror", "Occult Fiction"],
                                            "1/28/1977",
                                            "Jack Torrance’s new job at the Overlook Hotel is the perfect chance for a fresh start. As the off-season caretaker at the atmospheric old hotel, he’ll have plenty of time to spend reconnecting with his family and working on his writing. But as the harsh winter weather sets in, the idyllic location feels ever more remote . . . and more sinister. And the only one to notice the strange and terrible forces gathering around the Overlook is Danny Torrance, a uniquely gifted five-year-old.."
                                            );

    const id_1 = book_1._id;

    await reviewData.addReview(
                                "review1 title for shining",
                                "scaredycat",
                                5,
                                "10/7/2020",
                                "This book was creepy!!! It had me at the edge of my seat.  One of Stephan King's best work!",
                                id_1
                                )

    await reviewData.addReview(
                                "review2 title for shining",
                                "scaredycat",
                                5,
                                "10/7/2020",
                                "This book was creepy!!! It had me at the edge of my seat.  One of Stephan King's best work!",
                                id_1
                                )
                  

                                        
    const book_2 = await bookData.createBook( 
                                            "The Shining 2",
                                            "Stephen", 
                                            "King",
                                            ["Novel", "Horror fiction", "Gothic fiction", "Psychological horror", "Occult Fiction"],
                                            "1/28/2017",
                                            "Jack Torrance’s new job at the Overlook Hotel is the perfect chance for a fresh start. As the off-season caretaker at the atmospheric old hotel, he’ll have plenty of time to spend reconnecting with his family and working on his writing. But as the harsh winter weather sets in, the idyllic location feels ever more remote . . . and more sinister. And the only one to notice the strange and terrible forces gathering around the Overlook is Danny Torrance, a uniquely gifted five-year-old.."
                                            );
    const id_2 = book_2._id;

    const review_2_1 = await reviewData.addReview(
                            "review1 title for shining 2",
                            "scaredycat",
                            5,
                            "10/7/2020",
                            "This book was creepy!!! It had me at the edge of my seat.  One of Stephan King's best work!",
                            id_2
                            )

    const review_2_2 = await reviewData.addReview(
                            "review2 title for shining 2",
                            "scaredycat",
                            5,
                            "10/7/2020",
                            "This book was creepy!!! It had me at the edge of my seat.  One of Stephan King's best work!",
                            id_2
                            )


    console.log('Done seeding database');


    info = await bookData.removeBook(id_1);
    console.log(info);
                  
    
    const book_3 = await bookData.createBook( 
                                            "The Shining 3",
                                            "Stephen", 
                                            "King",
                                            ["Novel", "Horror fiction", "Gothic fiction", "Psychological horror", "Occult Fiction"],
                                            "1/28/2017",
                                            "Jack Torrance’s new job at the Overlook Hotel is the perfect chance for a fresh start. As the off-season caretaker at the atmospheric old hotel, he’ll have plenty of time to spend reconnecting with his family and working on his writing. But as the harsh winter weather sets in, the idyllic location feels ever more remote . . . and more sinister. And the only one to notice the strange and terrible forces gathering around the Overlook is Danny Torrance, a uniquely gifted five-year-old.."
                                            );
    const id_3 = book_3._id;

    const review_3_1 = await reviewData.addReview(
                            "review1 title for shining 3",
                            "scaredycat",
                            5,
                            "10/7/2020",
                            "This book was creepy!!! It had me at the edge of my seat.  One of Stephan King's best work!",
                            id_3
                            )

    const review_3_2 = await reviewData.addReview(
                            "review2 title for shining 3",
                            "scaredycat",
                            5,
                            "10/7/2020",
                            "This book was creepy!!! It had me at the edge of my seat.  One of Stephan King's best work!",
                            id_3
                            )



    const updatedBook_2 = {
                            "title": "The Shining UPDATED",
                            "author": {"authorFirstName": "Patrick", "authorLastName": "Hill"},
                            "genre": ["Novel", "Horror fiction", "Gothic fiction", "Psychological horror", "Occult Fiction"],
                            "datePublished": "1/28/1977",
                            "summary": "Jack Torrance’s new job at the Overlook Hotel is the perfect chance for a fresh start. As the off-season caretaker at the atmospheric old hotel, he’ll have plenty of time to spend reconnecting with his family and working on his writing. But as the harsh winter weather sets in, the idyllic location feels ever more remote . . . and more sinister. And the only one to notice the strange and terrible forces gathering around the Overlook is Danny Torrance, a uniquely gifted five-year-old.."
                            }
    const updatedBook = await bookData.updateBook(id_2, updatedBook_2 );



    // delete review on reviews
    const reviewsCollection = await reviews();
    await reviewsCollection.deleteOne({ _id: review_3_1._id });

    // delete review on Book
    const booksCollection = await books();
    const thatBooks = await booksCollection.find({"reviews": {$elemMatch:{_id : review_3_1._id}}}).toArray();
    const reviewdeleteinfo = await booksCollection.updateOne({ _id: thatBooks[0]._id }, { $pull: { reviews: {_id: review_3_1._id}} })

    // const thatReview = review_3_1
    // reviewdeleteinfo = await booksCollection.updateOne({ _id: thatBooks[0]._id }, { $pull: { reviews: thatReview} })



    await db.serverConfig.close();








    }

main();
