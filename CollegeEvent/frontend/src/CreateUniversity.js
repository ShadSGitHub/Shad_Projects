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
import { createUniversity } from "./store/actions/universityActions";

const CreateUniversity = ({ handleLogout }) => {
    const dispatch = useDispatch()
    const [input, setInput] = React.useState({ title: '', description: '' })

    const onSubmit = React.useCallback(e => {
        e.preventDefault()

        console.log(input)
        
        dispatch(createUniversity(input))
    }, [input, dispatch])

    const handleInputChange = React.useCallback(key => e => {
        
       const value = e?.target?.value ?? e

        setInput({ ...input, [key]: value })
    }, [input])

  return (
    <section className="CreateUniversity">
      <nav>
        <h2>Create University</h2>
      </nav>
      <div style={{ maxWidth: 520, margin: '16px auto' }}>
        <form onSubmit={onSubmit}>
          <div class="formBox">
            <label for="title">University Name</label>
            <input type="text" id="title" placeholder="University Title" value={input.title} onChange={handleInputChange('title')} />
          </div>
          <div class="formBox">
            <label for="Description">Description</label>
            <input type="text" id="Description" placeholder="Description"  value={input.description} onChange={handleInputChange('description')} />
          </div>
          <div class="formBox">
            <label for="Address">Address</label>
            <input type="text" id="Address" placeholder="Address"  />
          </div>
          <div class="formBox">
            <label for="students"># of Students</label>
            <input type="number" id="students" placeholder="students"  />
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

export default CreateUniversity;
