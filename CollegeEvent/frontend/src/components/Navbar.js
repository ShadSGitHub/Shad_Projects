import { useLocation, NavLink } from "react-router-dom";
import Select from "react-select";
import React from "react";


/*

	name of the app										role basis :: super admin | admin | student
	
	super admin: create university
	admin: create event
	student: create rso, view events

*/

const tabs = [
  [{ name: "Create University", route: "/CreateUniversity" }],
  [{ name: "Create Event", route: "/CreateEvent" }],
  [
    { name: "Create RSO", route: "/CreateRSO" },
    { name: "View Events", route: "/ViewEvents" },
  ],
];
const scopes = [
  { label: "Student", value: 2 },
  { label: "Admin", value: 1 },
  { label: "SuperAdmin", value: 0 },
];

function Navbar({ handleLogout, role }) {
  const[input, setInput] = React.useState({value: 0})
  console.log(input)
  return (

    <div className="navigation">
      <nav style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>Event Planning</h2>
        <button onClick={handleLogout}>Logout</button>
        <div class="formBox" >
            <label for="University">University</label>
            <Select options={scopes} value={input} onChange = {e => setInput(e)} />
        </div>
      </nav>
      <div className="navbar">
        {tabs[role || input.value].map((tab) => (
          <NavLink
            exact
            to={tab.route}
            className="main-nav"
            activeClassName="main-nav-active"
          >
            {tab.name}
          </NavLink>
        ))}
      </div>
    </div>
  );
}

export default Navbar;
