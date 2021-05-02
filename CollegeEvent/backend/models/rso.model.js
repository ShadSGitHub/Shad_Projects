const mongoose = require('mongoose');

const Schema = mongoose.Schema;
//    const [input, setInput] = React.useState({ title: '', description: '', university: null, rso: null, category: null, time: '', date: '', phone: '', email: '' , pricacy: null})
const rsoSchema = new Schema({
	title: {
		type: String,
		required: true,
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
	admin: {
		type: Object,
		required: true,
		trim: true
	},
	student1: {
		type: Object,
		required: true,
		trim: true
	},
	student2: {
		type: Object,
		required: true,
		trim: true
	},
	student3: {
		type: Object,
		required: true,
		trim: true
	},
	student4: {
		type: Object,
		required: true,
		trim: true
	},
	student5: {
		type: Object,
		required: true,
		trim: true
	},
}, {

	timestamps: true
});

const Rso = mongoose.model('Rso', rsoSchema);

module.exports = Rso;