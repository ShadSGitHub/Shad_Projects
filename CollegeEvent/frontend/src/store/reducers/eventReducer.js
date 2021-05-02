import types from '../types.json'

const eventReducer = (state={}, action) => {
	let newState = {}
	console.log(action, state)
	switch (action.type) {
		case types.EVENT_CREATE_SUCCESS:
			return { ...state, [action.payload.id]: action.payload }
		case types.EVENT_CREATE_FAILURE:
			return state
		case types.EVENTS_READ_SUCCESS:
			return action.payload
		case types.EVENTS_READ_FAILURE:
			return state
		case types.EVENT_UPDATE_SUCCESS:
			return { ...state, [action.payload.id]: action.payload }
		case types.EVENT_UPDATE_FAILURE:
			return state
		case types.EVENT_DELETE_SUCCESS:
			newState = { ...state }
			delete newState[action.payload]
			return newState
		case types.EVENT_DELETE_FAILURE:
			return state
		default:
			return state
	}
}

export default eventReducer
