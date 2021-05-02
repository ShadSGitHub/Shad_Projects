const router = require('express').Router();
let Rso = require('../models/rso.model');

router.route('/').get((req, res) => {
  Rso.find()
    .then(rsos => res.json(rsos))
    .catch(err => res.status(400).json('Error: ' + err));
});
//    const [input, setInput] = React.useState({ title: '', description: '', university: null, rso: null, category: null, time: '', date: '', phone: '', email: '' , pricacy: null})
router.route('/add').post((req, res) => {
  const title = req.body.title;
  const description = req.body.description;
  const student1 = req.body.student1;
  const student2 = req.body.student2;
  const student3 = req.body.student3;
  const student4 = req.body.student4;
  const student5 = req.body.student5;
  const admin = req.body.admin;
  const university = req.body.university;


  const newRso = new Rso({title, description, university, admin, student1, student2, student3, student4, student5});

  newRso.save()
    .then(() => res.json('Rso added!'))
    .catch(err => res.status(400).json('Error: ' + err));
});

router.route('/:id').get((req, res) => {
  Rso.findById(req.params.id)
    .then(rsos => res.json(rsos))
    .catch(err => res.status(400).json('Error: ' + err));
});

router.route('/:id').delete((req, res) => {
  Rso.findByIdAndDelete(req.params.id)
    .then(() => res.json('Rso deleted.'))
    .catch(err => res.status(400).json('Error: ' + err));
});

router.route('/update/:id').post((req, res) => {
  Rso.findById(req.params.id)
    .then(rsos => {
      rsos.title = req.body.title;
      rsos.description = req.body.description;
      rsos.student1 = req.body.student1;
      rsos.student2 = req.body.student2;
      rsos.student3 = req.body.student3;
      rsos.student4 = req.body.student4;
      rsos.student5 = req.body.student5;
      rsos.admin = req.body.admin;
      rsos.university = req.body.university;


      rsos.save()
        .then(() => res.json('Rso updated!'))
        .catch(err => res.status(400).json('Error: ' + err));
    })
    .catch(err => res.status(400).json('Error: ' + err));
});

module.exports = router;