<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
	  "http://www.w3.org/TR/html4/loose.dtd"> 
<html>
  <?php require("alian.php.inc"); ?>
  <head>
    <title>PPI Prediction</title>
    <meta http-equiv="Content-Type" content="text/html;charset=UTF-8" />
    <script type="text/javascript" src="jquery.min.js"></script>
    <script type="text/javascript" src="alian.js"></script>
	<script src="http://d3js.org/d3.v3.js"></script>
    <?php echo_drupal();  ?>
    <script type="text/javascript" src="./colorbox/colorbox/jquery.colorbox.js"></script>
    <link rel="stylesheet" type="text/css" href="colorbox.css" />
    <link rel="stylesheet" type="text/css" href="style.css" />
   <script>
   $(document).ready(function(){
       $(".ajax").colorbox({iframe:true, width:"1300px", height:"80%"});
     });
   $(document).ready(function() {
       $(".small-pdb").colorbox();
     });
    </script>
  </head>
  <body onload="adh()">
    <?php echo_udhead(); ?>
    <?php echo_quick(); ?>
	<?php echo_morris(); ?>
    <div class="whole" id="whole">
   <?php echo_head(); echo '<div id="progress"><img src="images/progress.gif" /> <br />Please wait a moment</div>'; flush();?>
      <div class="content">
	
	<?php 
   	  if(!empty($_POST['DDI']))
	    {
	      $user = $_SERVER["REMOTE_ADDR"];
	      $path = "/homes/annotation/htdocs/ppi_prediction/upload/";
	      $dir = $path . $user."/";

	      ppi_cleardir($dir);

	      $outputFileName = $dir."ddiOutput.txt";
	      $output = file_get_contents($outputFileName);
	      
	      $id = explode("_int_", $_POST['DDI']);
	      
	      $ddi = substr($_POST["DDI"],1);
	      
	      $file = $dir.'chosenDDI.txt';
	      $writeChosen = file_put_contents($file, $_POST["DDIid"]);
	      $file = $dir.'chosenDDI2.txt';
	      $writeChosen = file_put_contents($file, "query\n");
	      $writeChosen = file_put_contents($file, $ddi."\n", FILE_APPEND);
	      $writeChosen = file_put_contents($file, "1.5\n", FILE_APPEND);
	      
	      echo '<table><tr><td style="width:900px;"><h2>DDI between '.substr($id[0],1)." and " . $id[1] . ' is selected for prediction</h2></td><td><div id="tutorial"><a href="javascript:void(0);" id="help_button">Help</a></div></td></tr></table>';
	      echo '<div id="org_box"><span class="org_bot_cor"></span>The red parts in sequences are the domains, and the rest are in grey. </div>';
	      $d1 = substr($id[0],1);
	      $d2 = $id[1];

	      $cmd1 = 'export MATLAB_PREFDIR=/opt/MATLAB/;/usr/local/bin/matlab -nodesktop -nosplash -r "cd(\'/homes/annotation/htdocs/ppi_prediction/programs/\');predict(\''.$dir.'\');exit;"';

	      $cmd2 = 'export MATLAB_PREFDIR=/opt/MATLAB/;/usr/local/bin/matlab -nodesktop -nosplash -r "cd(\'/homes/annotation/htdocs/ppi_prediction/programs/\');drawQuery(\''.$dir.'\');exit;"';

	      $cmd3 = 'export MATLAB_PREFDIR=/opt/MATLAB/;/usr/local/bin/matlab -nodesktop -nosplash -r "cd(\'/homes/annotation/htdocs/ppi_prediction/programs/\');writehtml(\''.$dir.'\');exit;"';

	      shell_exec($cmd1);
	      shell_exec($cmd2);
	      shell_exec($cmd3);
	    }
$cmd = "chmod 777 $dir/*";
exec($cmd);
$cmd = "chmod 777 $dir/";
exec($cmd);   	  
   	  
   	  $rowCount = ($_POST["DDIid"] - 1)*5;
	  
	  $pieces = explode("\n", $output);
	  echo '<div class="step3">';
	  for ($i=1; $i<2; $i++) 
	    {
	      $IDs = explode("_int_", $pieces[$rowCount]);
	      echo "<table>";
	      echo "<tr valign=\"top\">";
	      $DDIid = 1 + floor($rowCount/5);
	      
	      echo "<td>";
	      
	      $headLine = explode(" ", $pieces[$rowCount+1]);
	      
	      $id = substr($headLine[2], 5, 7);
	      
	      echo '<div class="step3-seq">'."Domain <a href=\"http://pfam.sanger.ac.uk/family/".$id."\">".substr($IDs[0],1)."</a>";
	      echo " ("."<a href=\"http://3did.irbbarcelona.org/cgi-bin/query_pfamID_cgi.pl?draur=grahp&id=".$id."\">link to 3DID</a>".")<br/>";
	      echo ">".str_replace("Pfam", "Sequence Similarity score",$pieces[$rowCount + 1]);
	      echo "<br/>";
	      //Using &#8203; here allows the browser to break the line of text, without the visual distraction.
	      // $wraped = wordwrap($pieces[$rowCount + 2], 60, "\n", true);
	      $total = strlen($pieces[$rowCount + 2]);
	      
	      $count=floor($total/60);
	      $initC=0;       //define initial capital position;
	      $endC=$initC;   //define ending capital position;
	      $wraped2=$total + 7*$count;
	      
	      $wraped = wordwrap($pieces[$rowCount + 2], 60, "\n", true);
	      $has_cap = strpbrk($wraped, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'); // need to check
	      
	      $total = strlen($wraped);
	      $arr = str_split($wraped);
	      for ($j=$total; $j>$has_cap; $j--) 
		{
		  $b= strpbrk($arr[$j],'ABCDEFGHIJKLMNOPQRSTUVWXYZ');
		  if ( $b!=FALSE)
		    {
		      //echo "position $i: "."$b"."<br/>";
		      $endC=$j;
		      break;
		    };
		};
	      
	      $cap_cut = strlen($has_cap);
	      $lower_cut = $total-$endC;
	      
	      $temp = substr_replace($wraped, '<div class="dint">', $total - $cap_cut, 0);
	      $temp = substr_replace($temp, "</div>", -$lower_cut+1, 0); // need to correct for other possibilites
		  
	      
	      $br = str_replace("\n", "<br/>", $temp);

	      $br = $br. "</div></td>";
		  echo $br;
		  
		  $str_in_red_left = strip_tags(extractStrBetweenStrs($br, '<div class="dint">', "</div>"));
		  
		  
	      echo "<td valign=\"middle\">";
	      
	      echo '<img src="images/arrows.png" class="arrow" />';
	      echo "</td>";
	      echo "<td>";
	      
	      $headLine = explode(" ", $pieces[$rowCount+3]);
	      
	      $id = substr($headLine[2], 5, 7);
	      
	      echo '<div class="step3-seq">'."Domain <a href=\"http://pfam.sanger.ac.uk/family/".$id."\">".$IDs[1]."</a>";
	      echo " ("."<a href=\"http://3did.irbbarcelona.org/cgi-bin/query_pfamID_cgi.pl?draur=grahp&id=".$id."\">link to 3DID</a>".")<br/>";	      
	      echo ">".str_replace("Pfam","Sequence Similarity score",$pieces[$rowCount + 3]); 
	      echo "<br/>";

	      $total = strlen($pieces[$rowCount + 4]);	      
	      $count=floor($total/60);
	      $initC=0;       //define initial capital position;
	      $endC=$initC;   //define ending capital position;
	      $wraped2=$total + 7*$count;
	      
	      $wraped = wordwrap($pieces[$rowCount + 4], 60, "\n", true);
	      $has_cap = strpbrk($wraped, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'); // to check
	      //$tail_lower = strpbrk($has_cap, 'abcdefghijklmnopqrstuvwxyz'); // to check
	      
	      $total = strlen($wraped);
	      $arr = str_split($wraped);
	      for ($j=$total; $j>$has_cap; $j--) 
		{
		  $b= strpbrk($arr[$j],'ABCDEFGHIJKLMNOPQRSTUVWXYZ');
		  if ( $b!=FALSE)
		    {
		      $endC=$j;
		      break;
		    }
		}
	      
	      $cap_cut = strlen($has_cap);
	      $lower_cut = $total-$endC;
	      $temp = substr_replace($wraped, '<div class="dint">', $total - $cap_cut, 0);
	      $temp = substr_replace($temp, "</div>", -$lower_cut+1, 0);
	      
	      $br = str_replace("\n", "", $temp);
	      $br = str_replace("\r", "", $temp);
	      echo '<div class="wint">'.$br.'</div>';
		  
		  $str_in_red_right = strip_tags(extractStrBetweenStrs($br, '<div class="dint">', "</div>"));
		  
	      echo "</div></td></tr>";
	      if($apps = ppi_GetPredict($dir, $d1, $d2))
		echo '<tr><td colspan="2">'.$apps[0][0]."</td><td>".$apps[1][1]."</td></tr>";
	    }
	  echo '</table>';
	  
	  //show contact matrix
	     //**$outputFileName = $dir."/yourfilename.txt";
	     //pass information to matlab script
		 $cmd = '/usr/local/bin/matlab -nodesktop -nosplash -nojvm -r "cd(\'/homes/annotation/htdocs/ppi_prediction/\');test_Feb27(\''.$dir.'\', \''.substr($IDs[0],1).'\',\''.$str_in_red_left.'\', \''.$IDs[1].'\',\''. $str_in_red_right.'\');exit;" 2>&1';
         //**shell_exec($cmd);
		 //$dir = '/homes/annotation/htdocs/ppi_prediction/upload/128.4.113.36/'; //**for test only
		 //read contact pairs list from file, each record contains three numbers: position in sequence A, position in sequence B, score
		 $outputFileName = $dir."/test2_Feb27.txt"; //**this line only for testing
		 $output = file_get_contents($outputFileName);
		 $pieces = explode("\n", $output);
		 

		 $rowCount=0;
		 while(isset($pieces[$rowCount]) && strcmp($pieces[$rowCount], "") != 0)
	      {
		       
		       $singleRecord = explode(" ", $pieces[$rowCount]);
			   $data[] = array('SeqA' => $singleRecord[0], 'SeqB' => $singleRecord[1], 'Score' => $singleRecord[2]);
			   $data1[] = array('SeqA' => $singleRecord[0], 'SeqB' => $singleRecord[1], 'Score' => $singleRecord[2]);
			   $data2[] = array('SeqA' => $singleRecord[0], 'SeqB' => $singleRecord[1], 'Score' => $singleRecord[2]);
		       $rowCount++;
		  }
		  // sort pairs by score
		  foreach ($data as $key => $row) {
               $Score[$key]  = $row['Score'];
          }
		  // sort by SeqA
		  foreach ($data1 as $key => $row) {
               $SeqA[$key]  = $row['SeqA'];
          }
		  // sort by SeqB
		  foreach ($data2 as $key => $row) {
               $SeqB[$key]  = $row['SeqB'];
          }
		  array_multisort($Score, SORT_DESC, $data); //sorted by score
		  array_multisort($SeqA, SORT_ASC, $data1); //sorted by seqA
		  array_multisort($SeqB, SORT_ASC, $data2); //sorted by seqB
		  
		  
		  //only for test, print the list
		  //echo $cmd;
		  echo '<div id="sum_table">';
		  echo '<hr/><br/>';
          echo '<h4 style="display:inline;">Contact matrix: </h4><br/>';
          //**print_contactpairs($data);
	      echo '</div>';
		  echo '<div id = "chart"></div>
		  <script type="text/javascript" src="heatmap.js"></script>
	      ';
	  
if(file_exists($dir.'zscoreforweb'))
  $zscore = file_get_contents($dir.'zscoreforweb');
else
  $zscore = "No zscore";
	?>
      </div>
      <hr /><br />
      <h4 style="display:inline;">DDiFerSS prediction Z-Score: <?php echo $zscore ?></h4><?php echo_blanks(4); ?><button id="toggle_plot" type="button" value="Prediction Plot" class="round">Prediction Plot</button>
<div id="zscoreint"> <img src="./images/help.png" align="left" id="zhelp" /> A Z-Score greater than 3 indicates a reliable positive prediction. Compare with Zscores of known positive interacting pairs in the detailed results.</div>
	<div id="ppi_prediction" style="display:none;">
	  <?php
  if(file_exists("./upload/".$user."/predict.jpeg"))
    {
   $newfile = "http://annotation.dbi.udel.edu/ppi_prediction/upload/".$user."/predict.jpeg";
   echo '<IMG src="'.$newfile.'" border="0" width="850">';
    }
else
  echo 'No Predict Plot';
  
    
	
   ?>
	<h4>NOTE:</h4>
	This boxplot shows the
	distance to the hyperplane of the query sequence pair (green bullet)
	in a distribution of 100 random sequence pairs. Outliers are indicated
	by red crosses. 
      </div>
	<div id="sum_table">
	  <h4>Leave one out cross-validation result</h4>

	  <table id="sum_inner"><tr><th>No.</th><th>PDB ID</th><th>Seq 1</th><th>Seq 2</th><th>Z Score</th><th>Visualization</th><th>PDB Image</th><th>Interface Topology</th></tr>
	<?php
	       
	      $summary_dir = '/home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/dom_dom_ints/'.$d1."_int_" . $d2.'/summaryTable.txt'; 
	      if(file_exists($summary_dir))
		{
		 $summary = file_get_contents($summary_dir);
		 $summary_rows =  preg_split( '/\r\n|\r|\n/', $summary);
		 $i=1;
		 foreach ($summary_rows as $row)
		   {
		     $elements = preg_split('/\t|\s/ ',$row);
		     if($elements[5]!='')
		       {
			 $pfam1 = substr($elements[4],0,4);
			 $pfam2 = strtoupper(substr($elements[5],0,4));
			 $elements[4] = substr($elements[4], 4);
			 $elements[5] = substr($elements[5], 4);
			 $big_link = 'http://pir.georgetown.edu/pirwww/images/pdb/'.$pfam2.'x500c.jpg';
			 $small_link = '<a href="'.$big_link.'" class="small-pdb"><img src="http://pir.georgetown.edu/pirwww/images/pdb/'.$pfam2.'x25c.jpg" alt="Not Found" onerror="imgerror(this)"/></a>';
			 $pfam1 = '<a href="http://www.pdb.org/pdb/explore/explore.do?structureId='.$pfam1 .'" targe="_blank">'.$pfam1.'</a>';
			 //			 $elements[4] = str_replace($pfam1,'<a href="http://www.pdb.org/pdb/explore/explore.do?structureId='.$pfam1 .'" targe="_blank">'.$pfam1.'</a>',$elements[4]);
			 //			 $elements[5] = str_replace($pfam2,'<a href="http://www.pdb.org/pdb/explore/explore.do?structureId='.$pfam2 .'" target="_blank">'.$pfam2.'</a>',$elements[5]);
			 if($i % 2 == 0)
			   echo '<tr class="even"><td>Pair '.$i.'</td><td>'.$pfam1.'</td><td>'.$elements[4].'</td><td>'.$elements[5].'</td><td>'.$elements[7].'</td><td><a class="ajax" href="http://annotation.dbi.udel.edu/ppi_prediction/3d.php?pair='.$i.'&file='.$d1.'_int_'.$d2.'">more details</a></td><td>'.$small_link.' </td><td class="topology"></td></tr>';
			 else
			   echo '<tr class="odd"><td>Pair '.$i.'</td><td>'.$pfam1.'</td><td>'.$elements[4].'</td><td>'.$elements[5].'</td><td>'.$elements[7].'</td><td><a class="ajax" href="http://annotation.dbi.udel.edu/ppi_prediction/3d.php?pair='.$i.'&file='.$d1.'_int_'.$d2.'">more details</a></td><td>'.$small_link.'</td><td class="topology"></td></tr>';
		       }	     
		     $i++;
		   }
		}
	   ?>
	</table></div>
    </div>
</div>
<?php echo_foot(); ?>
</body>
</html>


