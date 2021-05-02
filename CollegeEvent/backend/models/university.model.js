const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const userSchema = new Schema({
	title: {
		type: String,
		required: true,
		unique: true,
		trim: true
	},
	description: {
		type: String,
		required: true,
		trim: true
	},
	
}, {

	timestamps: true
});

const University = mongoose.model('University', userSchema);

module.exports = University;