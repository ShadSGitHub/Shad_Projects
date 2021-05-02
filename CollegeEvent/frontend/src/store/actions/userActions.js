import axios from 'axios'
import types from '../types.json'

export const createUser = data => dispatch => {
	return axios({
	method: 'post',
	url: `http://localhost:5000/users/add`,
	data
	}).then(resp => {
	dispatch({ type: types.USER_CREATE_SUCCESS, payload: data })
	return Promise.resolve(`Created event ${resp.data.number}.`)
	}).catch(error => {
	dispatch({ type: types.USER_CREATE_FAILURE, error })
	return Promise.reject({ msg: 'An error has occurred.', error })
	})
}

export const readUser = () => dispatch => {
	return axios({
		method: 'get',
		url: `http://localhost:5000/users/:id`
	}).then((resp) => {
		const data = resp.data.reduce((o,r) => {
			o[r.id] = r
			return o
		}, {})

		dispatch({ type: types.USERS_READ_SUCCESS, payload: data })
		return Promise.resolve('event read | all good')
	})
	.catch(error => {
		dispatch({ type: types.USERS_READ_FAILURE, error })
		return Promise.reject({ msg: 'event read | everything is on fire', error })
	})
}

export const updateUser = data => dispatch => {
	return axios({
		method: 'put',
		url: `http://localhost:5000/users/update/:id`,
		data
	})
		.then(() => {
			dispatch({ type: types.USER_UPDATE_SUCCESS, payload: data })
			return Promise.resolve(`Updated event ${data.number}.`)
		})
		.catch(error => {
			dispatch({ type: types.USER_UPDATE_FAILURE, error })
			return Promise.reject({ msg: 'An error has occurred.', error })
		})
}

export const deleteUser = id => dispatch => {
	return axios({
		method: 'delete',
		url: `http://localhost:5000/users/:id`,
	}).then(resp => {
		dispatch({ type: types.USER_DELETE_SUCCESS, payload: id })
		return Promise.resolve(`Deleted event ${resp.data}.`)
	})
	.catch(error => {
		dispatch({ type: types.USER_DELETE_FAILURE, error })
		return Promise.reject({ msg: 'An error has occurred.', error })
	})
}
