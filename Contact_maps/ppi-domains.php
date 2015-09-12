<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
	  "http://www.w3.org/TR/html4/loose.dtd"> 
<html>
  <?php require("alian.php.inc"); ?>
  <head>
    <title>PPI Prediction</title>
    <meta http-equiv="Content-Type" content="text/html;charset=UTF-8" />
    <script type="text/javascript" src="jquery.min.js"></script>
    <script type="text/javascript" src="alian.js"></script>
    <?php echo_drupal(); ?>
	<?php echo_morris(); ?>
    <link rel="stylesheet" type="text/css" href="style.css" />
  </head>
  <body onload="adh()">
    <?php echo_udhead(); ?>
    <?php echo_quick(); ?>
    <div class="whole" id="whole">
   <?php echo_head(); 
         echo '<div id="progress"><img src="images/progress.gif" alt="Please wait a moment" /> <br />Please wait a moment</div>'; 
		 flush();
		 ?>
      <div class="content">	
	<?php 
	  $user = $_SERVER["REMOTE_ADDR"];
	  $path = "/homes/annotation/htdocs/ppi_prediction/upload/";
	  //$path = "C:/Users/Sun/Dropbox/wamp/www/PPI/upload/";
	  $dir = $path . $user;
	  //echo $user;
	  //echo $dir;
	  if(!is_dir($dir))
	    {
	      mkdir($dir);
	    }
   	  if(strcmp(substr($_POST["input_sequence1"], 0, 1), ">") != 0)
	    $seqA = ">seqA\n" . $_POST["input_sequence1"] . "\n";
	  else
	    $seqA = $_POST["input_sequence1"]."\n";
	  if(strcmp(substr($_POST["input_sequence2"], 0, 1), ">") != 0)
	    $seqB = ">seqB\n" . $_POST["input_sequence2"];
	  else
	    $seqB = $_POST["input_sequence2"];
	  $input_sequence = $seqA . $seqB;

	  $inputFileName = $dir."/requestfile_par.fasta";
	  $outputFileName = $dir."/ddiOutput.txt";
	  
          $evalue = $_POST["evalue"];

	  if(!empty($_FILES['uploadedfile']['name']))
	    {
	      if(move_uploaded_file($_FILES['uploadedfile']['tmp_name'], $inputFileName)) 
		{
		  echo "The file ".  basename( $_FILES['uploadedfile']['name'])." has been uploaded";
		}
	      else
		{
		  echo "There was an error uploading the file, please try again!";
		}
	    }
	    else
	      {                 	  
		file_put_contents($inputFileName,$input_sequence);
	      }

      shell_exec('/home/du/Protein_Protein_Interaction_Project/HMMER/hmmer-3.0-linux-intel-x86_64/binaries/hmmscan -E ' . $evalue . ' --domE ' . $evalue . ' --tblout '.$dir.'/hmmerOutputTab.txt /home/du/Protein_Protein_Interaction_Project/PFAM/Pfam-A.hmm '.$dir.'/requestfile_par.fasta > '.$dir.'/hmmerOutput.txt');
	  
	  //$cmd = 'export MATLAB_PREFDIR=/opt/MATLAB/;/opt/MATLAB/bin/matlab -nodesktop -nosplash -nojvm -r "cd(\'/homes/annotation/htdocs/ppi_prediction/programs/\');parsegetddi(\''.$dir.'/\');exit;" 2>&1';
	  $cmd = '/usr/local/bin/matlab -nodesktop -nosplash -nojvm -r "cd(\'/homes/annotation/htdocs/ppi_prediction/programs/\');parsegetddi(\''.$dir.'/\');exit;" 2>&1';

	 
shell_exec($cmd);
$output = file_get_contents($outputFileName);
$cmd = "chmod 777 $dir/*";
exec($cmd);
$cmd = "chmod 777 $dir/";
exec($cmd);
	  $pieces = explode("\n", $output);
	  //print_r($pieces);
	  $rowCount = 0;
	  echo '<table><tr><td style="padding-right:700px;"><h2>Interacting Domains</h2></td><td><div id="tutorial"><a href="javascript:void(0);" id="help_button">Help</a></div></td></tr></table>';
      echo '<div id="org_box"><span class="org_bot_cor"></span>The images are graphical view of the input sequences. The domains are in red, mapped via <a href="http://hmmer.janelia.org/" target="_blank">HMMER 3.0</a>, the rest are in grey.</div>';
	  echo '<div class="step2"><table id="domain-table" align="center">';
	  $bar_name = 0;

	    while(isset($pieces[$rowCount]) && strcmp($pieces[$rowCount], "") != 0)
	      {
		if($rowCount % 5 == 0)
		  {
		    $display_flag = 1;
		    $IDs = explode("_int_", $pieces[$rowCount]);
		    $DDIid = 1 + floor($rowCount/5); 
		    
		    $ddi_row = "";
		    $ddi_row .= '<tr class="seqtr" id="'.$DDIid.'">';
		    $ddi_row .= '<td valign="middle" align="center" width="10%"><h3>'.$DDIid."</h3></td>";
		    $ddi_row .= '<td align="center" width="30%"><div class="step2-seq">';
		    
		    $headLine = explode(" ", $pieces[$rowCount+1]);
		    $id = substr($headLine[2], 5, 7);

		    $ddi_row .= "Domain <a href=\"http://pfam.sanger.ac.uk/family/".$id."\">".substr($IDs[0],1)."</a>";
		    $ddi_row .= " ("."<a href=\"http://3did.irbbarcelona.org/cgi-bin/query_pfamID_cgi.pl?draur=grahp&id=".$id."\">link to 3DID</a>".")<br/>";

		    preg_match("'e=(.+?) '",$pieces[$rowCount + 1],$ematch);
		    if($evalue < $ematch[1])
		      $display_flag = 0;
		    //		    echo number_format($ematch[1],100,'.','');

		    $ddi_row .= ">".$pieces[$rowCount + 1]; 

		    //Using &#8203; here allows the browser to break the line of text, without the visual distraction. 
		    $wraped = wordwrap($pieces[$rowCount + 2], 60, "\n", true);
		    $total = strlen($pieces[$rowCount + 2]);

		    $count=floor($total/60);
		    $initC=0;	//define initial capital position;
		    $endC=$initC;	//define ending capital position;
		    $wraped2=$total + 7*$count;	

		    //$wraped = wordwrap($pieces[$rowCount + 2], 60, "&#8203;", true);
		    $has_cap = strpbrk($wraped, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'); // need to check 
		    //$has_cap = stripos($wraped,$cap_string);
		    $total = strlen($wraped);
		    $arr = str_split($wraped);
		    $cap_start = stripos($wraped, $has_cap);
		    for ($i=$total-1; $i>$cap_start; $i--) 
		      {
			if((ord($arr[$i])<92) && (ord($arr[$i]) > 64))
			  {
			    $endC=$i;
			    break;
			  }
		      };
		    
		    //$cap_cut = strlen($has_cap);
		    //$lower_cut = $total-$endC;
		    
		    //$final = substr($wraped,0,$cap_start).'<div class="dint">'.substr($wraped, $cap_start, $endC-$cap_start+1)."</div>".substr($wraped, $endC+1,$total-$endC-1);
		    
		    $cap_start_1 = $cap_start;
		    $cap_end_1 = $endC;
		    $total_1 = $total;
		    

		    		    
		    $br = str_replace("\n", "<br/>", $final);
		    
		    $headLine = explode(" ", $pieces[$rowCount+3]);
		    //echo "headLine: $headLine[2] <br/>";
		    $id = substr($headLine[2], 5, 7);
		    //echo "id : $id <br/>";
		    $total = strlen($pieces[$rowCount + 4]);

		    $count=floor($total/60);
		    $initC=0;       //define initial capital position;
		    $endC=$initC;   //define ending capital position;
		    $wraped2=$total + 7*$count;
		    
		    $wraped = wordwrap($pieces[$rowCount + 4], 60,  "\n", true);
		    $has_cap = strpbrk($wraped, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'); // to check
		    $total = strlen($wraped);		  
		    $arr = str_split($wraped);
		    $cap_start = stripos($wraped, $has_cap);

		    for ($i=$total; $i>$cap_start; $i--) 
		      {
			if((ord($arr[$i])<92) && (ord($arr[$i]) > 64))
			  {
			    $endC=$i;
			    break;
			  }
		      };
		    
		    //$cap_cut = strlen($has_cap);
		    //$lower_cut = $total-$endC;
		    //$final = substr($wraped,0,$cap_start).'<div class="dint">'.substr($wraped, $cap_start, $endC-$cap_start+1)."</div>".substr($wraped, $endC+1,$total-$endC-1);		    
		    $br = str_replace("\n", "<br/>", $final);
		    
		    $cap_start_2 = $cap_start;
		    $cap_end_2 = $endC;
		    $total_2 = $total;
		    
		    if($total_1 > $total_2)
		      {
			$per_1 = 1;
			$per_2 = $total_2/$total_1;
			
		      }
		    else
		      {
			$per_1 = $total_1/$total_2;
			$per_2 = 1;
		      }
		    //		    echo "<br />".$per_1." ".$per_2;
		    ////ppi_getinterbar($dir."/",$cap_start_1/$total_1,$cap_end_1/$total_1, $per_1,$bar_name);
		    ////$ddi_row .= '<img src="./upload/'.$user."/".$bar_name.'.png"></img>';

		    $ddi_row .= "</div></td>";
		    $ddi_row .= '<td valign="middle">';

		    $ddi_row .= '<img src="images/arrows.png" class="arrow" />'; 
		    $ddi_row .= "</td>";
		    $ddi_row .= '<td class="step2-seq" width="30%">';

		    $ddi_row .= "Domain <a href=\"http://pfam.sanger.ac.uk/family/".$id."\">".$IDs[1]."</a>";
		    $ddi_row .= " ("."<a href=\"http://3did.irbbarcelona.org/cgi-bin/query_pfamID_cgi.pl?draur=grahp&id=".$id."\">link to 3DID</a>".")<br/>";
		    
		    $ddi_row .= ">".$pieces[$rowCount + 3]; 

		    preg_match("'e=(.+?) '",$pieces[$rowCount + 3],$ematch);
		    if($evalue < $ematch[1])
		      $display_flag = 0;

		    $bar_name++;

		    ////ppi_getinterbar($dir."/", $cap_start_2/$total_2,$cap_end_2/$total_2, $per_2,$bar_name);
			
		    ////$ddi_row .= '<img src="./upload/'.$user."/".$bar_name.'.png"></img>';
		    $bar_name++;
		    
		    //echo $br;
		    
		    //echo '</td></tr><tr class="step2-form"><td colspan="3">';		   
		    $ddi_row .= '</td><td>';
		    $ddi_row .= '<form method="post" action="ppi-details.php" id="'.$DDIid.'" style="width:100px;">';
		    $ddi_row .= '<input type="hidden" name="DDIid" value="'.$DDIid.'">';
		    $ddi_row .= '<input type="hidden" name="DDI" value="'.$pieces[$rowCount].'">';
		    $ddi_row .= '<input type="submit" value="Use this DDI" class="round"></input>';
		    $ddi_row .= "</form></td></tr>";
		    if($display_flag)
		      echo $ddi_row;
		    $rowCount += 5;
		  }                         
	      }
	    echo '</table></div>';	    
	?>
<br />

<!-- only for test of morris.js-->
<hr />
<div id="donut-example" style="height: 250px;"></div>
<?php firstgraph(); ?>
<!---->

<div>
	<FORM METHOD="LINK" ACTION="ppi.php" style="display:inline;float:left;">
	  <INPUT TYPE="submit" VALUE="Start new ppi prediction" class="round">
	</FORM>
	      <a class="domain-download" href="./upload/<?php echo $user.'/hmmerOutput.txt'; ?>" target="_blank">Download list of PFAM domains mapped</a> 
<a class="domain-download" href="./upload/<?php echo $user.'/ddiOutput.txt'; ?>" target="_blank">Download list of interactions</a>
</div>
<br />
<br />
<br />
      </div>
    </div>
    <?php echo_foot(); ?>
  </body>
</html>
