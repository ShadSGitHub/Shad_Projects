import React from 'react';
import {BrowserRouter as Router, Route, Switch, NavLink, Redirect} from "react-router-dom";
import { DropdownButton, Dropdown } from "react-bootstrap";
import Select from "react-select";

import { useDispatch } from "react-redux";
import { useSelector } from 'react-redux';
import { readEvents } from "./store/actions/eventActions";
const GrabEvents = () =>
{
  	const [events] = useSelector(state => [state.events])
  	          	console.log(events)

  	return(
  		<table className="table">
          <thead className="thead-light">
            <tr>
              <td>Events</td>
              <th>Description</th>
              <th>University</th>
              <th>RSO</th>
              <th>Category</th>
              <th>Time</th>
              <th>Date</th>
              <th>Phone</th>
              <th>Email</th>
              <th>Privacy</th>
            </tr>
          </thead>
          <tbody>
          {Object.keys(events).map(k=> {
          	const event = events[k]
          	return(
          		<tr key={k}>
          			<th>{event.title}</th>
          			<th>{event.description}</th>
          			<th>{event.university.value}</th>
          			<th>{event.rso.value}</th>
          			<th>{event.category.value}</th>
          			<th>{event.time}</th>
          			<th>{event.date}</th>
          			<th>{event.phone}</th>
          			<th>{event.email}</th>
          			<th>{event.privacy.value}</th>
          		</tr>
          		)
          	})}
          </tbody>
          
        </table>
        )
}

const ViewEvents = ({handleLogout}) => {
	const dispatch = useDispatch()

  // Initializes events by calling this on app startup
  	React.useEffect(() => {
    	let promises = [dispatch(readEvents())]
    	Promise.all(promises)
      	.then(() => {
      })
      .catch(() => {
      })
  }, [dispatch])

return(
	<section className="ViewEvents">
	<div>
        <nav><h2>Events List</h2></nav>
        <GrabEvents />
      </div>
    </section>
	)

}

export default ViewEvents;