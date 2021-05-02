import types from '../types.json'

const userReducer = (state={}, action) => {
	let newState = {}
	console.log(action, state)
	switch (action.type) {
		case types.USER_CREATE_SUCCESS:
			return { ...state, [action.payload.id]: action.payload }
		case types.USER_CREATE_FAILURE:
			return state
		case types.USERS_READ_SUCCESS:
			return action.payload
		case types.USERS_READ_FAILURE:
			return state
		case types.USER_UPDATE_SUCCESS:
			return { ...state, [action.payload.id]: action.payload }
		case types.USER_UPDATE_FAILURE:
			return state
		case types.USER_DELETE_SUCCESS:
			newState = { ...state }
			delete newState[action.payload]
			return newState
		case types.USER_DELETE_FAILURE:
			return state
		default:
			return state
	}
}

export default userReducer
