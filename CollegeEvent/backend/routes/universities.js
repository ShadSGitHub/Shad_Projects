const router = require('express').Router();
let University = require('../models/university.model');

router.route('/').get((req, res) => {
  University.find()
    .then(universities => res.json(universities))
    .catch(err => res.status(400).json('Error: ' + err));
});
//    const [input, setInput] = React.useState({ title: '', description: '', university: null, rso: null, category: null, time: '', date: '', phone: '', email: '' , pricacy: null})
router.route('/add').post((req, res) => {
  const title = req.body.title;
  const description = req.body.description;

  const newUniversity = new University({title, description});

  newUniversity.save()
    .then(() => res.json('University added!'))
    .catch(err => res.status(400).json('Error: ' + err));
});

router.route('/:id').get((req, res) => {
  University.findById(req.params.id)
    .then(universities => res.json(universities))
    .catch(err => res.status(400).json('Error: ' + err));
});

router.route('/:id').delete((req, res) => {
  University.findByIdAndDelete(req.params.id)
    .then(() => res.json('University deleted.'))
    .catch(err => res.status(400).json('Error: ' + err));
});

router.route('/update/:id').post((req, res) => {
  University.findById(req.params.id)
    .then(universities => {
      universities.title = req.body.title;
      universities.description = req.body.description;
      


      universities.save()
        .then(() => res.json('University updated!'))
        .catch(err => res.status(400).json('Error: ' + err));
    })
    .catch(err => res.status(400).json('Error: ' + err));
});

module.exports = router;