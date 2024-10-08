//CAPTCHA GENERATING FUNCTION
function cap() {
    document.getElementById("submit").disabled = false;
  var alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V'
    , 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
    'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '!', '@', '#', '$', '%', '^', '&', '*', '+'];
  var a = alpha[Math.floor(Math.random() * 71)];
  var b = alpha[Math.floor(Math.random() * 71)];
  var c = alpha[Math.floor(Math.random() * 71)];
  var d = alpha[Math.floor(Math.random() * 71)];
  var e = alpha[Math.floor(Math.random() * 71)];
  var f = alpha[Math.floor(Math.random() * 71)];

  var final = a + b + c + d + e + f;
  document.getElementById("capt").value = final;
}

// Captcha Validation Function
function validcap() {
  var stg1 = document.getElementById('capt').value;
  var stg2 = document.getElementById('textinput').value;

  if (stg1 === stg2) {
      // Enables login button if captcha is valid
      document.getElementById("submit").disabled = false;
      return true;
  } else {
      document.getElementById("submit").disabled = true;
      document.getElementById('invalidcap').value = "invalid";

      // Show a custom modal or alert dialog
      var confirmation = confirm("Please enter a valid captcha. Click OK to refresh the page.");

      // Refresh the page if user clicks OK
      if (confirmation) {
          location.reload(); // Refresh the page
      }
      return false;
  }
}
