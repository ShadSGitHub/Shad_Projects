import React from "react";
import {
  BrowserRouter as Router,
  Route,
  Switch,
  Link,
  Redirect,
} from "react-router-dom";
import { DropdownButton, Dropdown } from "react-bootstrap";
import Select from "react-select";

import { useDispatch } from "react-redux";
import { createRSO } from "./store/actions/RSOActions";

const Universities = [
  { label: "UCF", value: "UCF" },
  { label: "FSU", value: "FSU" },
];
const Admin = [
  { label: "ipleauwps@gmail.com", value: "ipleauwps@gmail.com" },
  { label: "shadysaleh@gmail.com", value: "shadysaleh@gmail.com" },

];
const Student = [
  { label: "Ipleau@knights.ucf.edu", value: "ipleau@knights.ucf.edu" },
  { label: "test@gmail.com", value: "test@gmail.com" },
  { label: "Student3", value: "Student3" },
  { label: "Student4", value: "Student4" },
  { label: "Student5", value: "Student5" },
  { label: "Student6", value: "Student6" },
];

const CreateRSO = ({ handleLogout }) => {
    const dispatch = useDispatch()
    const [input, setInput] = React.useState({ title: '', description: '', university: '', admin: '', student1: '', student2: '', student3: '', student4: '', student5: ''})

    const onSubmit = React.useCallback(e => {
        e.preventDefault()

        console.log(input)
        
        dispatch(createRSO(input))
    }, [input, dispatch])

    const handleInputChange = React.useCallback(key => e => {
        
       const value = e?.target?.value ?? e

        setInput({ ...input, [key]: value })
    }, [input])

  return (
    <section className="CreateRSO">
      <nav>
        <h2>Create RSO</h2>
      </nav>
      <div style={{ maxWidth: 520, margin: '16px auto' }}>
        <form onSubmit={onSubmit}>
          <div class="formBox">
            <label for="title">RSO Title</label>
            <input type="text" id="title" placeholder="RSO Description" value={input.title} onChange={handleInputChange('title')} />
          </div>
          <div class="formBox">
            <label for="Description">Description</label>
            <input type="text" id="Description" placeholder="Description" value={input.description} onChange={handleInputChange('description')} />
          </div>
          <div class="formBox">
            <label for="University">University</label>
            <Select options={Universities} value={input.university} onChange={handleInputChange('university')} />
          </div>
          <div class="formBox">
            <label for="Admin">Admin</label>
            <Select options={Admin} value={input.admin} onChange={handleInputChange('admin')} />
          </div>
          <div class="formBox">
            <label for="Student">Student 1</label>
            <Select options={Student} value={input.Student} onChange={handleInputChange('student1')} />
          </div>
          <div class="formBox">
            <label for="Student">Student 2</label>
            <Select options={Student} value={input.Student} onChange={handleInputChange('student2')} />
          </div>
          <div class="formBox">
            <label for="Student">Student 3</label>
            <Select options={Student} value={input.Student} onChange={handleInputChange('student3')} />
          </div>
          <div class="formBox">
            <label for="Student">Student 4</label>
            <Select options={Student} value={input.Student} onChange={handleInputChange('student4')} />
          </div>
          <div class="formBox">
            <label for="Student">Student 5</label>
            <Select options={Student} value={input.Student} onChange={handleInputChange('student5')} />
          </div>
          
          <div class="formBox">
            <button id="btn" type='submit'>Click to Add</button>
          </div>
          <div id="msg">
            <pre></pre>
          </div>
        </form>
      </div>
    </section>
  );
};

export default CreateRSO;
