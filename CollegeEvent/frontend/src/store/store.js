import { createStore, applyMiddleware } from 'redux'
import thunk from 'redux-thunk'
import axios from 'axios'

import rootReducer from './reducers/rootReducer'

axios.interceptors.request.use(config => {
	console.log('request sent', config.url)
	console.log(config)
	
	return config
})

const store = createStore(
	rootReducer,
	applyMiddleware(thunk)
)

export default store
