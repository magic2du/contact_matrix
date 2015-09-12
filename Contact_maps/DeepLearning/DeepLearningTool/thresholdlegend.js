
    var svg2 = d3.select("#threshold_legend")
	    .append("svg")
			.attr("width", width2 + margin2.left + margin2.right)
			.attr("height", 100)
        .append("g");

    
	var legend2 = svg2.selectAll(".legend2")
	          .append("g")
              .attr("class", "legend2");

          legend2.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", legendElementWidth2)
            //.attr("height", gridSize / 2)
			.attr("height", 20)
            .style("fill", colors2[0]);
			
		  legend2.append("rect")
            .attr("x", 200)
            .attr("y", 0)
            .attr("width", legendElementWidth2)
            //.attr("height", gridSize / 2)
			.attr("height", 20)
            .style("fill", colors2[1]);

          legend2.append("text")
            .attr("class", "mono")
			.text("< "+"threshold value")
            .attr("x", 0)
            //.attr("y", height + gridSize);
			.attr("y", 40);	
			
		  legend2.append("text")
            .attr("class", "mono")
			.text("¡Ý "+"threshold value")
            .attr("x", 200)
            //.attr("y", height + gridSize);
			.attr("y", 40);	

