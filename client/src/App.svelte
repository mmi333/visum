<script>
  import { Input, Image, Button } from 'sveltestrap';
  import autosize from 'svelte-autosize';

  let text;
  let image;
  let summary;

  function upload() {
      const formData = new FormData();
      formData.append('text', text);
      const upload = fetch('http://127.0.0.1:5000/upload', {
          method: 'POST',
          body: formData
      }).then((response) => response.json()).then((result) => {
          console.log('Success:', result);
          summary = result.answer;
          image = true;
      })
              .catch((error) => {
                  console.error('Error:', error);
              });
  }
</script>
<style>
body {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    width: 100%;
    background-color: #212529;
}
:global(.main-container) {
    display: flex;
    background: #212529;
    align-items: center;
    justify-content: center;
    height: 100%;
    width: 100%;

}


:global(.flex-container) {
  display: flex;
}

:global(.container) {
  display: flex;
  justify-content: center;
  align-items: center;
}
:global(.text-container) {
  display: flex;
  justify-content: center;
  align-items: center;
}
textarea {
    display: flex;

}


</style>
<head>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/css/bootstrap.min.css">
  </head>
<body>
<div class="main-container">
    <div class="text-container" style="text-align: center">

        <textarea use:autosize id="textSender" type="text" bind:value={text} cols="100" />

    </div>
    <div class="flex-container">

        <div class="container" style="text-align: center">


        {#if image}

        <Image src="images/out.png" alt="background image" />
        {/if}

        {#if summary}

        <p>
            {summary}
        </p>
        {/if}
        <Button style="align-items: center;" class="submit-button" on:click={upload}>Submit</Button>
        </div>

    </div>

</div>
</body>