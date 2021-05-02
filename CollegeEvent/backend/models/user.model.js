const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const userSchema = new Schema({
	uid: {
		type: String,
		required: true,
		unique: true,
		trim: true
	},
	email: {
		type: String,
		required: true,
		unique: true,
		trim: true
	},
	scopes: {
		type: String,
		required: true,
		trim: true
	},
	rso: {
		type: String,
		required: true,
		trim: true
	},
	university: {
		type: String,
		required: true,
		trim: true
	},
}, {

	timestamps: true
});

const User = mongoose.model('User', userSchema);

module.exports = User;