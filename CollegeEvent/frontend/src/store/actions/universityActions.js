import axios from 'axios'
import types from '../types.json'

export const createUniversity = data => dispatch => {
	return axios({
	method: 'post',
	url: `http://localhost:5000/universities/add`,
	data
	}).then(resp => {
	dispatch({ type: types.UNIVERSITY_CREATE_SUCCESS, payload: data })
	return Promise.resolve(`Created event ${resp.data.number}.`)
	}).catch(error => {
	dispatch({ type: types.UNIVERSITY_CREATE_FAILURE, error })
	return Promise.reject({ msg: 'An error has occurred.', error })
	})
}

export const readUniversity = () => dispatch => {
	return axios({
		method: 'get',
		url: `http://localhost:5000/universities/:id`,
	}).then((resp) => {
		const data = resp.data.reduce((o,r) => {
			o[r.id] = r
			return o
		}, {})

		dispatch({ type: types.UNIVERSITIES_READ_SUCCESS, payload: data })
		return Promise.resolve('event read | all good')
	})
	.catch(error => {
		dispatch({ type: types.UNIVERSITIES_READ_FAILURE, error })
		return Promise.reject({ msg: 'event read | everything is on fire', error })
	})
}

export const updateUniversity = data => dispatch => {
	return axios({
		method: 'put',
		url: `http://localhost:5000/universities/update/:id`,
		data
	})
		.then(() => {
			dispatch({ type: types.UNIVERSITY_UPDATE_SUCCESS, payload: data })
			return Promise.resolve(`Updated event ${data.number}.`)
		})
		.catch(error => {
			dispatch({ type: types.UNIVERSITY_UPDATE_FAILURE, error })
			return Promise.reject({ msg: 'An error has occurred.', error })
		})
}

export const deleteUniversity = id => dispatch => {
	return axios({
		method: 'delete',
		url: `http://localhost:5000/universities/:id`,
	}).then(resp => {
		dispatch({ type: types.UNIVERSITY_DELETE_SUCCESS, payload: id })
		return Promise.resolve(`Deleted event ${resp.data}.`)
	})
	.catch(error => {
		dispatch({ type: types.UNIVERSITY_DELETE_FAILURE, error })
		return Promise.reject({ msg: 'An error has occurred.', error })
	})
}
