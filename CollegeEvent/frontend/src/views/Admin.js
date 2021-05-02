import React from 'react';
import {BrowserRouter as Router, Route, Switch, NavLink, Redirect} from "react-router-dom";
import Select from "react-select";

const RSO = [
  { label: "UCF", value: "UCF" },
  { label: "FSU", value: "FSU" },
];

const Universities = [
  { label: "UCF", value: "UCF" },
  { label: "FSU", value: "FSU" },
];

const Admin = ({handleLogout}) => {
	return(
		<section className = "Admin">
			<nav>
				<h2>Admin Event Planning</h2>
				<button onClick={handleLogout}> Logout </button>
			</nav>
			<body>
				<NavLink exact to="/CreateEvent" className="main-nav" activeClassName="main-nav-active">Create Event</NavLink>
				        <form>
          <div class="formBox">
            <label for="RSO">RSO</label>
            <Select options={RSO} />
          </div>
          <div class="formBox">
            <button id="btn">Set RSO</button>
          </div>
          <div id="msg">
            <pre></pre>
          </div>
        </form>
        <form>
          <div class="formBox">
            <label for="University">Select University</label>
            <Select options={Universities} />
          </div>
          <div class="formBox">
            <button id="btn">Set University</button>
          </div>
          <div id="msg">
            <pre></pre>
          </div>
        </form>
			</body>
		</section>
		)

}

export default Admin;