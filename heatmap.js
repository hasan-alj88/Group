// Aurther : Bryce York
// Source : codepen.io/bryceyork/pen/mtqAl

function clean_formatted_data(str) {
  return parseFloat(str.replace(/([%,$,\,])+/g,''));
}

function col_to_array(tbl_col,target) {
  // Returns column `n` (zero indexed) in table id `target` as an array

  var colArray = $('#'+target+' td:nth-child('+tbl_col+')').map(function(){
    return clean_formatted_data( $(this).text() );
  }).get();

  return colArray;
}

//------ new schtuff ------------------------//

function get_pos_of_max(col_data) { return $.inArray( Math.max.apply(Math,col_data), col_data ) }

function generate_opacities(col_data, max) {
  var opacity_array = [];
  var increment = max/(col_data.length);

  for(i=col_data.length; i >= 1; i--) {
    opacity_array.push(i*increment/100);
  }

  return opacity_array;
}


function process_col_best_performing(tbl_col, target) {
  var col_data = col_to_array(tbl_col,target);
  var opacity_array = generate_opacities(col_data, 50);
  var row_count = col_data.length;

  for (var i=1; i <= row_count; i++) {
    $('#'+target+' tr:nth-child('+(get_pos_of_max(col_data)+1)+') td:nth-child('+tbl_col+')').css('background','rgba(0,0,255,'+opacity_array[0]+')');
    col_data[get_pos_of_max(col_data)] = null;
    opacity_array.splice(0,1);

  }
}

process_col_best_performing(2,'js-datatable');
process_col_best_performing(3,'js-datatable');
process_col_best_performing(4,'js-datatable');
process_col_best_performing(5,'js-datatable');