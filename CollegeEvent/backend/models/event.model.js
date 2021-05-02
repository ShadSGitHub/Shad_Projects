const mongoose = require('mongoose');

const Schema = mongoose.Schema;
//    const [input, setInput] = React.useState({ title: '', description: '', university: null, rso: null, category: null, time: '', date: '', phone: '', email: '' , pricacy: null})
const eventSchema = new Schema({
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
	university: {
		type: Object,
		required: true,
		trim: true
	},
	rso: {
		type: Object,
		required: true,
		trim: true
	},
	category: {
		type: Object,
		required: true,
		trim: true
	},
	time: {
		type: String,
		required: true,
		trim: true
	},
	date: {
		type: Date,
		required: true,
		trim: true
	},
	phone: {
		type: String,
		required: true,
		trim: true
	},
	email: {
		type: String,
		required: true,
		trim: true
	},
	privacy: {
		type: Object,
		required: true,
		trim: true
	},

}, {

	timestamps: true
});

const Event = mongoose.model('Event', eventSchema);

module.exports = Event;