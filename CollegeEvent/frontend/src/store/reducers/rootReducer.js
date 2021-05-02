import { combineReducers } from 'redux'

import eventReducer from './eventReducer'
import rsoReducer from './rsoReducer'
import universityReducer from './universityReducer'
import userReducer from './userReducer'


const rootReducer = combineReducers({
	events: eventReducer,
	rso: rsoReducer,
	university: universityReducer,
	user: userReducer,
})

export default rootReducer
