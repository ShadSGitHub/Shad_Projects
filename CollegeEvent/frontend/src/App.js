import logo from "./logo.svg";
import React, { useState, useEffect } from "react";
import "./assets/css/App.css";
import fire from "./fire";
import Login from "./Login";
import LandingPage from "./LandingPage";
import Admin from "./views/Admin";
import SuperAdmin from "./views/SuperAdmin";
import Student from "./views/Student";
import CreateEvent from "./CreateEvent";
import CreateRSO from "./CreateRSO";
import ViewEvents from "./ViewEvents";
import CreateUniversity  from "./CreateUniversity";
import { DropdownButton, Dropdown } from "react-bootstrap";
import Select from "react-select";

import { useDispatch } from "react-redux";
import { createUser } from "./store/actions/userActions";


import Navbar from './components/Navbar'
import { Provider } from "react-redux";
import store from "./store/store";

import {
  BrowserRouter as Router,
  Route,
  Link,
  useHistory,
} from "react-router-dom";


const App = () => {
  const dispatch = useDispatch()
  const [user, setUser] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [emailError, setEmailError] = useState("");
  const [passwordError, setPasswordError] = useState("");
  const [hasAccount, setHasAccount] = useState(false);
  

  const clearInputs = () => {
    setEmail("");
    setPassword("");
  };

  const clearErrors = () => {
    setEmailError("");
    setPasswordError("");
  };

  const handleLogin = () => {
    clearErrors();
    fire
      .auth()
      .signInWithEmailAndPassword(email, password)
      .catch((err) => {
        switch (err.code) {
          case "auth/invalid-email":
          case "auth/user-disabled":
          case "auth/user-not-found":
            setEmailError(err.message);
            break;
          case "auth/wrong-password":
            setPasswordError(err.message);
            break;
          default:
            break;
        }
      });
    history.push("/LandingPage");
  };
  const history = useHistory();

  const handleHistory = () => {
    history.push("login");
  };
  const handleSignup = () => {
    clearErrors();
    fire
      .auth()
      .createUserWithEmailAndPassword(email, password)
      .then((user) => {
        /* user handler */
        dispatch(createUser({uid: user.user.uid, email: email, scopes: 'student', rso: 'none', university: 'none'}))

      })
      .catch((err) => {
        switch (err.code) {
          case "auth/email-already-in-use":
          case "auth/invalid-email":
            setEmailError(err.message);
            break;
          case "auth/weak-password":
            setPasswordError(err.message);
            break;
          default:
            break;
        }
      });
  };

  const handleLogout = () => {
    fire.auth().signOut();
    history.push("/");
  };

  React.useEffect(() => {
    const unsubscribe = fire.auth().onAuthStateChanged((user) => {
      if (user) {
        clearInputs();
        setUser(user);
      } else {
        setUser("");
      }
    });
    
    return () => {
      unsubscribe()
    }
  }, [])
  return (

    <>
      <Navbar handleLogout={handleLogout} />

        <Route exact path="/LandingPage">
          <LandingPage handleLogout={handleLogout} />
        </Route>
        <Route exact path="/CreateEvent">
          <CreateEvent />
        </Route>
        <Route exact path="/CreateRSO">
          <CreateRSO />
        </Route>
         <Route exact path="/CreateUniversity">
          <CreateUniversity />
        </Route>
        <Route exact path="/ViewEvents">
          <ViewEvents />
        </Route>
        <Route exact path="/">
          <Login
            email={email}
            setEmail={setEmail}
            password={password}
            setPassword={setPassword}
            handleLogin={handleLogin}
            handleSignup={handleSignup}
            hasAccount={hasAccount}
            setHasAccount={setHasAccount}
            emailError={emailError}
            passwordError={passwordError}
          />
        </Route>
        <Route exact path="/Admin">
          <Admin handleLogout={handleLogout} />
        </Route>
        <Route exact path="/SuperAdmin">
          <SuperAdmin handleLogout={handleLogout} />
        </Route>
        <Route exact path="/Student">
          <Student handleLogout={handleLogout} />
        </Route>
    </>
  );
};

export default App;
