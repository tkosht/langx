// For authoring Nightwatch tests, see
// http://nightwatchjs.org/guide#usage

module.exports = {

  'default_e2e_tests': function (browser) {
    // automatically uses dev Server port from /config.index.js
    // default: http://localhost:8080
    // see nightwatch.conf.js
    const path = require('path')
    const devServer = browser.globals.devServerURL
    const test_csv_file = path.resolve(__dirname, "test.csv") // int the same directory as this js script

    browser
      .url('http://localhost:8000')
      .waitForElementVisible('#app', 5000)
      .assert.elementPresent('.row')
      .waitForElementVisible('#upload_form', 1000)
      .assert.elementCount('img', 1)
      .waitForElementPresent('textarea#textArea.form-control', 100)
      .assert.urlContains('http://localhost:8000')
      .assert.urlEquals('http://localhost:8000/front/index.html#/')
      .setValue('input#upload_file', test_csv_file)
      .assert.value('textarea#textArea.form-control', 'Loaded samples in the head of file')
      .saveScreenshot('./screenshot/' + browser.currentTest.name + '_01_loaded_csv.png')
      .click('button#submit.btn.btn-dark')
      .assert.elementPresent('div.progress')
      .saveScreenshot('./screenshot/' + browser.currentTest.name + '_02_submitted.png')
      .waitForElementNotPresent('div.progress', 300000)
      .saveScreenshot('./screenshot/' + browser.currentTest.name + '_03_done.png')
      .assert.value('textarea#textArea.form-control', "success: to upload the file 'test.csv'")
      // .useXpath()
      .end()
  }
}

function setElementText (selector, value) {
  document.querySelector(selector).textContent = value
}
