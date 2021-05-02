const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');

require('dotenv').config();

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

const uri = process.env.ATLAS_URI;
mongoose.connect(uri, { useNewUrlParser: true, useCreateIndex: true }
);
const connection = mongoose.connection;
connection.once('open', () => {
  console.log("MongoDB database connection established successfully");
})

const userRouter = require('./routes/users');
const eventRouter = require('./routes/events');
const universityRouter = require('./routes/universities');
const rsoRouter = require('./routes/rsos');
app.use('/rsos', rsoRouter)
app.use('/universities', universityRouter)
app.use('/users', userRouter)
app.use('/events', eventRouter)


app.listen(port, () => {
    console.log(`Server is running on port: ${port}`);
});