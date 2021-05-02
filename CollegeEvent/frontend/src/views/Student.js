import React from "react";
import {
  BrowserRouter as Router,
  Route,
  Switch,
  NavLink,
  Redirect,
} from "react-router-dom";
import Select from "react-select";

import { useDispatch } from "react-redux";
import { updateUser } from "../store/actions/userActions";
const RSO = [
  { label: "UCF", value: "UCF" },
  { label: "FSU", value: "FSU" },
];

const Universities = [
  { label: "UCF", value: "UCF" },
  { label: "FSU", value: "FSU" },
];

const Student = ({ handleLogout }) => {
  const dispatch = useDispatch()
    const [input, setInput] = React.useState({rso: null, university: null})

    const onSubmit = React.useCallback(e => {
        e.preventDefault()

        console.log(input)
        
        dispatch(updateUser(input))
    }, [input, dispatch])

    const handleInputChange = React.useCallback(key => e => {
        
       const value = e?.target?.value ?? e

        setInput({ ...input, [key]: value })
    }, [input])

  return (
    <section className="Student">
      <nav>
        <h2>Student Event Planning</h2>
        <button onClick={handleLogout}> Logout </button>
      </nav>
      <body>
        <NavLink
          exact
          to="/CreateRSO"
          className="main-nav"
          activeClassName="main-nav-active"
        >
          Create RSO
        </NavLink>
        <NavLink
          exact
          to="/ViewEvents"
          className="main-nav"
          activeClassName="main-nav-active"
        >
          View Events
        </NavLink>

        <form onSubmit = {onSubmit}>
          <div class="formBox">
            <label for="RSO">RSO</label>
            <Select options={RSO} value={input.rso} onChange={handleInputChange('rso')} />          
          </div>
          <div class="formBox">
            <button id="btn" type = 'submit'>Set RSO</button>
          </div>
          <div id="msg">
            <pre></pre>
          </div>
        </form>
        <form onSubmit = {onSubmit}>
          <div class="formBox">
            <label for="University">Select University</label>
            <Select options={Universities} value={input.university} onChange={handleInputChange('university')} />          
            </div>
          <div class="formBox">
            <button id="btn" type='submit'>Set University</button>
          </div>
          <div id="msg">
            <pre></pre>
          </div>
        </form>
      </body>
    </section>
  );
};

export default Student;
