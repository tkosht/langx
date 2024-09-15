import Vue from 'vue'
import Router from 'vue-router'
import HelloWorld from '@/components/HelloWorld'
import FileInput from '@/components/FileInput'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/upload',
      name: 'FileInput',
      component: FileInput
    },
    {
      path: '/',
      name: 'HelloWorld',
      component: HelloWorld
    }
  ]
})
