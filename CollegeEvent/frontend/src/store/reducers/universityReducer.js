import types from '../types.json'

const universityReducer = (state={}, action) => {
	let newState = {}
	console.log(action, state)
	switch (action.type) {
		case types.UNIVERSITY_CREATE_SUCCESS:
			return { ...state, [action.payload.id]: action.payload }
		case types.UNIVERSITY_CREATE_FAILURE:
			return state
		case types.UNIVERSITIES_READ_SUCCESS:
			return action.payload
		case types.UNIVERSITIES_READ_FAILURE:
			return state
		case types.UNIVERSITY_UPDATE_SUCCESS:
			return { ...state, [action.payload.id]: action.payload }
		case types.UNIVERSITY_UPDATE_FAILURE:
			return state
		case types.UNIVERSITY_DELETE_SUCCESS:
			newState = { ...state }
			delete newState[action.payload]
			return newState
		case types.UNIVERSITY_DELETE_FAILURE:
			return state
		default:
			return state
	}
}

export default universityReducer	
