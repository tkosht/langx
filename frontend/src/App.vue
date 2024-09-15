<template>
  <div id="app" class="container">
    <img src="./assets/logo.png" /><br />
    <form-upload />
  </div>
</template>

<script>
import FormUpload from './components/FormUpload'
export default {
  name: 'App',
  components: {
    FormUpload
  }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
