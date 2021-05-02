const router = require('express').Router();
let Event = require('../models/event.model');

router.route('/').get((req, res) => {
  Event.find()
    .then(events => res.json(events))
    .catch(err => res.status(400).json('Error: ' + err));
});
//    const [input, setInput] = React.useState({ title: '', description: '', university: null, rso: null, category: null, time: '', date: '', phone: '', email: '' , pricacy: null})
router.route('/add').post((req, res) => {
  const title = req.body.title;
  const description = req.body.description;
  const time = req.body.time;
  const date = req.body.date;
  const phone = req.body.phone;
  const category = req.body.category;
  const email = req.body.email;
  const privacy = req.body.privacy;
  const rso = req.body.rso;
  const university = req.body.university;


  const newEvent = new Event({title, description, university, rso, category, time, date, phone, email, privacy});

  newEvent.save()
    .then(() => res.json('Event added!'))
    .catch(err => res.status(400).json('Error: ' + err));
});

router.route('/:id').get((req, res) => {
  Event.findById(req.params.id)
    .then(events => res.json(events))
    .catch(err => res.status(400).json('Error: ' + err));
});

router.route('/:id').delete((req, res) => {
  Event.findByIdAndDelete(req.params.id)
    .then(() => res.json('Event deleted.'))
    .catch(err => res.status(400).json('Error: ' + err));
});

router.route('/update/:id').post((req, res) => {
  Event.findById(req.params.id)
    .then(events => {
      events.title = req.body.title;
      events.description = req.body.description;
      events.time = req.body.time;
      events.email = req.body.email;
      events.privacy = req.body.privacy;
      events.rso = req.body.rso;
      events.university = req.body.university;
      events.category = req.body.category;
      events.date = req.body.date;
      events.phone = req.body.phone;


      events.save()
        .then(() => res.json('Event updated!'))
        .catch(err => res.status(400).json('Error: ' + err));
    })
    .catch(err => res.status(400).json('Error: ' + err));
});

module.exports = router;