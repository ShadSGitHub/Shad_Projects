import React from 'react';
import {BrowserRouter as Router, Route, Switch, NavLink, Redirect} from "react-router-dom";

const SuperAdmin = ({handleLogout}) => {
	return(
		<section className = "Super Admin">
			<nav>
				<h2>SuperAdmin Event Planning</h2>
				<button onClick={handleLogout}> Logout </button>

			</nav>
			<body>
				<NavLink exact to="/CreateUniversity" className="main-nav" activeClassName="main-nav-active">Create University</NavLink>
			</body>
		</section>
		)

}

export default SuperAdmin;