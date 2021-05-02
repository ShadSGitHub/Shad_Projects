import types from '../types.json'

const rsoReducer = (state={}, action) => {
	let newState = {}
	console.log(action, state)
	switch (action.type) {
		case types.RSO_CREATE_SUCCESS:
			return { ...state, [action.payload.id]: action.payload }
		case types.RSO_CREATE_FAILURE:
			return state
		case types.RSOS_READ_SUCCESS:
			return action.payload
		case types.RSOS_READ_FAILURE:
			return state
		case types.RSO_UPDATE_SUCCESS:
			return { ...state, [action.payload.id]: action.payload }
		case types.RSO_UPDATE_FAILURE:
			return state
		case types.RSO_DELETE_SUCCESS:
			newState = { ...state }
			delete newState[action.payload]
			return newState
		case types.RSO_DELETE_FAILURE:
			return state
		default:
			return state
	}
}

export default rsoReducer
