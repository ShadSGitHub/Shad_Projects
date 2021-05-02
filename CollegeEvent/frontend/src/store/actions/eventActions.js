import axios from 'axios'
import types from '../types.json'

export const createEvent = data => dispatch => {
	return axios({
	method: 'post',
	url: `http://localhost:5000/events/add`,
	data
	}).then(resp => {
	dispatch({ type: types.EVENT_CREATE_SUCCESS, payload: data })
	return Promise.resolve(`Created event ${resp.data.number}.`)
	}).catch(error => {
	dispatch({ type: types.EVENT_CREATE_FAILURE, error })
	return Promise.reject({ msg: 'An error has occurred.', error })
	})
}

export const readEvents = () => dispatch => {
	return axios({
		method: 'get',
		url: `http://localhost:5000/events/`,
	}).then((resp) => {


		dispatch({ type: types.EVENTS_READ_SUCCESS, payload: resp.data })
		return Promise.resolve('event read | all good')
	})
	.catch(error => {
		dispatch({ type: types.EVENTS_READ_FAILURE, error })
		return Promise.reject({ msg: 'event read | everything is on fire', error })
	})
}

export const updateEvent = data => dispatch => {
	return axios({
		method: 'put',
		url: `http://localhost:5000/events/update/:id`,
		data
	})
		.then(() => {
			dispatch({ type: types.EVENT_UPDATE_SUCCESS, payload: data })
			return Promise.resolve(`Updated event ${data.number}.`)
		})
		.catch(error => {
			dispatch({ type: types.EVENT_UPDATE_FAILURE, error })
			return Promise.reject({ msg: 'An error has occurred.', error })
		})
}

export const deleteEvent = id => dispatch => {
	return axios({
		method: 'delete',
		url: `http://localhost:5000/users/:id`,
	}).then(resp => {
		dispatch({ type: types.EVENT_DELETE_SUCCESS, payload: id })
		return Promise.resolve(`Deleted event ${resp.data}.`)
	})
	.catch(error => {
		dispatch({ type: types.EVENT_DELETE_FAILURE, error })
		return Promise.reject({ msg: 'An error has occurred.', error })
	})
}
