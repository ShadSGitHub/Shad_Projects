import firebase from 'firebase';

var firebaseConfig = {
    apiKey: "AIzaSyAWJVFlagtIig41Ea8VQWeKE9dkvs-2i2A",
    authDomain: "login-and-register-e1fd9.firebaseapp.com",
    projectId: "login-and-register-e1fd9",
    storageBucket: "login-and-register-e1fd9.appspot.com",
    messagingSenderId: "757877107028",
    appId: "1:757877107028:web:6d4d85f821616c9b6bc0c1"
};
const fire = firebase.initializeApp(firebaseConfig);

export default fire;